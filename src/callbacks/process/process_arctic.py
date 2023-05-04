import common.camera as camera
import common.data_utils as data_utils
import common.transforms as tf
import src.callbacks.process.process_generic as generic


def process_data(
    models, inputs, targets, meta_info, mode, args, field_max=float("inf")
):
    img_res = 224
    K = meta_info["intrinsics"]
    gt_pose_r = targets["mano.pose.r"]  # MANO pose parameters
    gt_betas_r = targets["mano.beta.r"]  # MANO beta parameters

    gt_pose_l = targets["mano.pose.l"]  # MANO pose parameters
    gt_betas_l = targets["mano.beta.l"]  # MANO beta parameters

    gt_kp2d_b = targets["object.kp2d.norm.b"]  # 2D keypoints for object base
    gt_object_rot = targets["object.rot"].view(-1, 3)

    # pose the object without translation (call it object cano space)
    out = models["arti_head"].object_tensors.forward(
        angles=targets["object.radian"].view(-1, 1),
        global_orient=gt_object_rot,
        transl=None,
        query_names=meta_info["query_names"],
    )
    diameters = out["diameter"]
    parts_idx = out["parts_ids"]
    meta_info["part_ids"] = parts_idx
    meta_info["diameter"] = diameters

    # targets keypoints of hand and objects are in camera coord (full resolution image) space
    # map all entities from camera coord to object cano space based on the rigid-transform
    # between the object base keypoints in camera coord and object cano space
    # since R, T is used, relative distance btw hand and object is preserved
    num_kps = out["kp3d"].shape[1] // 2
    kp3d_b_cano = out["kp3d"][:, num_kps:]
    R0, T0 = tf.batch_solve_rigid_tf(targets["object.kp3d.full.b"], kp3d_b_cano)
    joints3d_r0 = tf.rigid_tf_torch_batch(targets["mano.j3d.full.r"], R0, T0)
    joints3d_l0 = tf.rigid_tf_torch_batch(targets["mano.j3d.full.l"], R0, T0)

    # pose MANO in MANO canonical space
    gt_out_r = models["mano_r"](
        betas=gt_betas_r,
        hand_pose=gt_pose_r[:, 3:],
        global_orient=gt_pose_r[:, :3],
        transl=None,
    )
    gt_model_joints_r = gt_out_r.joints
    gt_vertices_r = gt_out_r.vertices
    gt_root_cano_r = gt_out_r.joints[:, 0]

    gt_out_l = models["mano_l"](
        betas=gt_betas_l,
        hand_pose=gt_pose_l[:, 3:],
        global_orient=gt_pose_l[:, :3],
        transl=None,
    )
    gt_model_joints_l = gt_out_l.joints
    gt_vertices_l = gt_out_l.vertices
    gt_root_cano_l = gt_out_l.joints[:, 0]

    # map MANO mesh to object canonical space
    Tr0 = (joints3d_r0 - gt_model_joints_r).mean(dim=1)
    Tl0 = (joints3d_l0 - gt_model_joints_l).mean(dim=1)
    gt_model_joints_r = joints3d_r0
    gt_model_joints_l = joints3d_l0
    gt_vertices_r += Tr0[:, None, :]
    gt_vertices_l += Tl0[:, None, :]

    # now that everything is in the object canonical space
    # find camera translation for rendering relative to the object

    # unnorm 2d keypoints
    gt_kp2d_b_cano = data_utils.unormalize_kp2d(gt_kp2d_b, img_res)

    # estimate camera translation by solving 2d to 3d correspondence
    gt_transl = camera.estimate_translation_k(
        kp3d_b_cano,
        gt_kp2d_b_cano,
        meta_info["intrinsics"].cpu().numpy(),
        use_all_joints=True,
        pad_2d=True,
    )

    # move to camera coord
    gt_vertices_r = gt_vertices_r + gt_transl[:, None, :]
    gt_vertices_l = gt_vertices_l + gt_transl[:, None, :]
    gt_model_joints_r = gt_model_joints_r + gt_transl[:, None, :]
    gt_model_joints_l = gt_model_joints_l + gt_transl[:, None, :]

    ####
    gt_kp3d_o = out["kp3d"] + gt_transl[:, None, :]
    gt_bbox3d_o = out["bbox3d"] + gt_transl[:, None, :]

    # roots
    gt_root_cam_patch_r = gt_model_joints_r[:, 0]
    gt_root_cam_patch_l = gt_model_joints_l[:, 0]
    gt_cam_t_r = gt_root_cam_patch_r - gt_root_cano_r
    gt_cam_t_l = gt_root_cam_patch_l - gt_root_cano_l
    gt_cam_t_o = gt_transl

    targets["mano.cam_t.r"] = gt_cam_t_r
    targets["mano.cam_t.l"] = gt_cam_t_l
    targets["object.cam_t"] = gt_cam_t_o

    avg_focal_length = (K[:, 0, 0] + K[:, 1, 1]) / 2.0
    gt_cam_t_wp_r = camera.perspective_to_weak_perspective_torch(
        gt_cam_t_r, avg_focal_length, img_res
    )

    gt_cam_t_wp_l = camera.perspective_to_weak_perspective_torch(
        gt_cam_t_l, avg_focal_length, img_res
    )

    gt_cam_t_wp_o = camera.perspective_to_weak_perspective_torch(
        gt_cam_t_o, avg_focal_length, img_res
    )

    targets["mano.cam_t.wp.r"] = gt_cam_t_wp_r
    targets["mano.cam_t.wp.l"] = gt_cam_t_wp_l
    targets["object.cam_t.wp"] = gt_cam_t_wp_o

    # cam coord of patch
    targets["object.cam_t.kp3d.b"] = gt_transl

    targets["mano.v3d.cam.r"] = gt_vertices_r
    targets["mano.v3d.cam.l"] = gt_vertices_l
    targets["mano.j3d.cam.r"] = gt_model_joints_r
    targets["mano.j3d.cam.l"] = gt_model_joints_l
    targets["object.kp3d.cam"] = gt_kp3d_o
    targets["object.bbox3d.cam"] = gt_bbox3d_o

    out = models["arti_head"].object_tensors.forward(
        angles=targets["object.radian"].view(-1, 1),
        global_orient=gt_object_rot,
        transl=None,
        query_names=meta_info["query_names"],
    )

    # GT vertices relative to right hand root
    targets["object.v.cam"] = out["v"] + gt_transl[:, None, :]
    targets["object.v_len"] = out["v_len"]

    targets["object.f"] = out["f"]
    targets["object.f_len"] = out["f_len"]

    targets = generic.prepare_interfield(targets, field_max)

    return inputs, targets, meta_info
