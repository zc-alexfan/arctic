# import pytorch3d.transforms as tf
import json
import os
import os.path as op
import sys

import numpy as np
import torch

import common.rot as rot
import common.transforms as tf

sys.path = ["."] + sys.path

import common.body_models as human_models
import common.ld_utils as ld_utils
import common.thing as thing
from common.ld_utils import cat_dl
from src.arctic.preprocess_dataset import construct_loader

with open("./data/arctic_data/data/meta/misc.json", "r") as f:
    misc = json.load(f)

IGNORE_KEYS = ["v_len", "bottom_anchor", "f", "f_len", "parts_ids", "mask", "diameter"]


def compute_bbox_batch(kp2d, obj_s):
    assert isinstance(kp2d, torch.Tensor)
    assert len(kp2d.shape) == 3
    # (batch, view, 2)
    x_max = kp2d[:, :, 0].max(dim=1).values
    x_min = kp2d[:, :, 0].min(dim=1).values

    y_max = kp2d[:, :, 1].max(dim=1).values
    y_min = kp2d[:, :, 1].min(dim=1).values

    x_dim = x_max - x_min
    y_dim = y_max - y_min

    obj_scale = torch.maximum(x_dim, y_dim) * obj_s

    bbox = torch.FloatTensor(get_bbox_from_kp2d(kp2d.cpu().numpy()).T).to(kp2d.device)

    cx = bbox[:, 0]
    cy = bbox[:, 1]
    bbox_w = bbox[:, 2]
    bbox_h = bbox[:, 3]
    bbox_dim = torch.maximum(bbox_w, bbox_h) + obj_scale

    scale = bbox_dim / 200.0
    center = [cx, cy]
    return scale, center


def forward_define_bbox(out_all_2d, obj_s):
    # statcams bbox
    kp2d = out_all_2d["verts.object"][:, :9]
    batch_size = kp2d.shape[0]
    kp2d = kp2d.reshape(batch_size * 9, -1, 2)

    scale, (cx, cy) = compute_bbox_batch(kp2d, obj_s)
    scale = scale.view(batch_size, 9)
    cx = cx.view(batch_size, 9)
    cy = cy.view(batch_size, 9)

    # egocam bbox: it has fixed dim
    ego_cx = 2800 / 2.0
    ego_cy = 2000 / 2.0
    ego_dim = 2800 / 200.0
    cx[:, 0] = ego_cx
    cy[:, 0] = ego_cy
    scale[:, 0] = ego_dim

    # smallest bbox with size 600px x 600px
    scale[:, 1:] = torch.clamp(scale[:, 1:], 3.0, None)
    bbox = torch.stack((cx, cy, scale), dim=2)
    return bbox


def process_batch(
    batch,
    layers,
    smplx_m,
    world2cam,
    intris_mat,
    image_sizes,
    sid,
    export_verts,
):
    out_world = forward_gt_world(batch, layers, smplx_m)
    out_all_views = forward_world2cam(batch, out_world, world2cam)
    out_pts_views = []
    for out in out_all_views:
        out_pts = {k: v for k, v in out.items() if "rot" not in k}
        out_pts_views.append(out_pts)
    out_all_2d = forward_project2d(batch, out_pts_views, intris_mat)
    bbox = forward_define_bbox(out_all_2d, obj_s=0.6)
    out_valid = forward_valid(
        bbox,
        out_all_2d["joints.right"],
        out_all_2d["joints.left"],
        out_all_2d["verts.object"],
        image_sizes,
        sid,
    )

    if not export_verts:
        # remove verts from wolrd
        keys = list(out_world.keys())
        _out_world = {}
        for key in keys:
            if "verts" in key:
                continue
            _out_world[key] = out_world[key]
        out_world = _out_world

    # remove not needed terms in out_all_views
    keys = list(out_all_views[0].keys())
    out_views = {}
    for key in keys:
        if not export_verts and "verts" in key:
            continue
        if "rot" in key:
            out_views[key] = torch.stack(
                [out_all_views[idx][key] for idx in range(len(out_all_views) - 1)],
                dim=1,
            )
        else:
            out_views[key] = torch.stack(
                [out_all_views[idx][key] for idx in range(len(out_all_views))], dim=1
            )

    out_views.update(out_valid)

    if not export_verts:
        # remove not needed terms in 2d
        out_all_2d = {k: v for k, v in out_all_2d.items() if "verts" not in k}
    return out_world, out_views, out_all_2d, bbox


def transform_mano_rot_cam(rot_r_world, world2cam):
    world2cam_batch = world2cam[:, :3, :3]
    quat_world2cam = rot.matrix_to_quaternion(world2cam_batch).cuda()
    rot_r_quat = rot.axis_angle_to_quaternion(rot_r_world)
    rot_r_cam = rot.quaternion_to_axis_angle(
        rot.quaternion_multiply(quat_world2cam, rot_r_quat)
    )
    return rot_r_cam


def transform_points_dict(world2cam, pts3d_dict):
    out_all_cam = {}
    for key, pts_world in pts3d_dict.items():
        if "rot" not in key and key not in IGNORE_KEYS:
            out_all_cam[key] = tf.transform_points_batch(world2cam, pts_world)
    rot_r_cam = transform_mano_rot_cam(pts3d_dict["rot_r"], world2cam)
    rot_l_cam = transform_mano_rot_cam(pts3d_dict["rot_l"], world2cam)
    obj_rot_cam = transform_mano_rot_cam(pts3d_dict["obj_rot"], world2cam)
    out_all_cam["rot_r_cam"] = rot_r_cam
    out_all_cam["rot_l_cam"] = rot_l_cam
    out_all_cam["obj_rot_cam"] = obj_rot_cam
    return out_all_cam


def project_2d_dict(K, pts_dict):
    # project all points in a dict via intrinsics
    # return dict
    out_2d = {}
    for key, pts3d_cam in pts_dict.items():
        out_2d[key] = tf.project2d_batch(K, pts3d_cam)
    return out_2d


def forward_gt_world(batch, layers, smplx_m):
    # world coord GT
    out = layers["right"](
        global_orient=batch["rot_r"],
        hand_pose=batch["pose_r"],
        betas=batch["shape_r"],
        transl=batch["trans_r"],
    )
    mano_r = {"verts.right": out.vertices, "joints.right": out.joints}

    out = layers["left"](
        global_orient=batch["rot_l"],
        hand_pose=batch["pose_l"],
        betas=batch["shape_l"],
        transl=batch["trans_l"],
    )
    mano_l = {"verts.left": out.vertices, "joints.left": out.joints}
    assert mano_l["joints.left"].shape[1] == 21

    # smplx
    params = {}
    params["global_orient"] = batch["smplx_global_orient"]
    params["body_pose"] = batch["smplx_body_pose"]
    params["left_hand_pose"] = batch["smplx_left_hand_pose"]
    params["right_hand_pose"] = batch["smplx_right_hand_pose"]
    params["jaw_pose"] = batch["smplx_jaw_pose"]
    params["leye_pose"] = batch["smplx_leye_pose"]
    params["reye_pose"] = batch["smplx_reye_pose"]
    params["transl"] = batch["smplx_transl"]

    out = smplx_m(**params)
    smplx_v = out.vertices
    smplx_j = out.joints

    smplx_out = {"verts.smplx": smplx_v, "joints.smplx": smplx_j}
    query_names = batch["query_names"]

    with torch.no_grad():
        obj_out = layers["object"](
            angles=batch["obj_arti"].view(-1, 1),
            global_orient=batch["obj_rot"],
            transl=batch["obj_trans"] / 1000,
            query_names=query_names,
        )  # vicon coord

        # v_sub
        # parts_sub_ids
        obj_out.pop("v_sub")
        obj_out.pop("parts_sub_ids")
        vo = obj_out["v"]
        obj_out.pop("v")
        obj_out["verts.object"] = vo

    out_all = {}
    out_all.update(mano_r)
    out_all.update(mano_l)
    out_all.update(obj_out)
    out_all.update(smplx_out)
    out_all["rot_r"] = batch["rot_r"]
    out_all["rot_l"] = batch["rot_l"]
    out_all["obj_rot"] = batch["obj_rot"]
    return out_all


def forward_world2cam(batch, out_world, world2cam):
    # [ego, 8 views, distort ego]
    batch_size = batch["world2ego"].shape[0]

    # egocentric view: undistorted space
    out_all_views = []
    out_all_views.append(transform_points_dict(batch["world2ego"], out_world))
    # allocentric views
    for view_idx in range(8):
        out_all_views.append(
            transform_points_dict(
                world2cam[view_idx : view_idx + 1].repeat(batch_size, 1, 1), out_world
            )
        )
    # egocentric view: distorted space
    out_all_cam = {}
    for key, pts_world in out_world.items():
        if "rot" in key or key in IGNORE_KEYS:
            continue

        pts3d_ego = tf.transform_points_batch(batch["world2ego"], pts_world)
        dist = batch["dist"][0]  # same within subject
        dist_pts = tf.distort_pts3d_all(pts3d_ego, dist)
        out_all_cam[key] = dist_pts
    out_all_views.append(out_all_cam)
    return out_all_views


def forward_project2d(batch, out_all_views, intris_mat):
    batch_size = batch["K_ego"].shape[0]

    # project 2d: ego undist
    out_all_2d = []
    out_all_2d.append(project_2d_dict(batch["K_ego"], out_all_views[0]))

    # project 2d: allocentric
    for view_idx in range(8):
        # ego, allo, allo, ..
        out_2d = project_2d_dict(
            intris_mat[view_idx : view_idx + 1].repeat(batch_size, 1, 1),
            out_all_views[view_idx + 1],
        )
        out_all_2d.append(out_2d)

    # project 2d: ego dist
    out_all_2d.append(project_2d_dict(batch["K_ego"], out_all_views[-1]))

    # reform tensors
    out_all_views = ld_utils.ld2dl(out_all_views)
    for key, out in out_all_views.items():
        out_all_views[key] = torch.stack(out, dim=1)

    out_all_2d = ld_utils.ld2dl(out_all_2d)
    for key, out in out_all_2d.items():
        out_all_2d[key] = torch.stack(out, dim=1)

    return out_all_2d


def get_bbox_from_kp2d(kp_2d):
    if len(kp_2d.shape) > 2:
        ul = np.array(
            [kp_2d[:, :, 0].min(axis=1), kp_2d[:, :, 1].min(axis=1)]
        )  # upper left
        lr = np.array(
            [kp_2d[:, :, 0].max(axis=1), kp_2d[:, :, 1].max(axis=1)]
        )  # lower right
    else:
        ul = np.array([kp_2d[:, 0].min(), kp_2d[:, 1].min()])  # upper left
        lr = np.array([kp_2d[:, 0].max(), kp_2d[:, 1].max()])  # lower right

    # ul[1] -= (lr[1] - ul[1]) * 0.10  # prevent cutting the head
    w = lr[0] - ul[0]
    h = lr[1] - ul[1]
    c_x, c_y = ul[0] + w / 2, ul[1] + h / 2
    # to keep the aspect ratio
    w = h = np.where(w / h > 1, w, h)
    w = h = h * 1.1

    bbox = np.array([c_x, c_y, w, h])  # shape = (4,N)
    return bbox


def bbox_jts_to_valid(bboxes, j2d):
    # scale is de-normalized
    assert isinstance(bboxes, torch.Tensor)
    assert isinstance(j2d, torch.Tensor)
    assert bboxes.shape[0] == j2d.shape[0]
    assert j2d.shape[1] == 9
    assert bboxes.shape[1] == 9
    xmin = bboxes[:, :, 0]
    ymin = bboxes[:, :, 1]
    xmax = bboxes[:, :, 2]
    ymax = bboxes[:, :, 3]

    xvalid = (xmin[:, :, None] <= j2d[:, :, :, 0]) * (
        j2d[:, :, :, 0] <= xmax[:, :, None]
    )

    yvalid = (ymin[:, :, None] <= j2d[:, :, :, 1]) * (
        j2d[:, :, :, 1] <= ymax[:, :, None]
    )

    jts_valid = xvalid * yvalid

    jts_valid = jts_valid.long()
    return jts_valid


def forward_valid(bboxes, joints_right, joints_left, verts_object, image_sizes, sid):
    view_ind = np.arange(9)
    view_ind[0] = 9
    j2d_r = joints_right[:, view_ind].clone()
    j2d_l = joints_left[:, view_ind].clone()
    v2d_o = verts_object[:, view_ind].clone().mean(dim=2)[:, :, None, :]
    dev = v2d_o.device

    # prepare bboxes
    im_sizes = torch.FloatTensor(np.array(image_sizes[sid])).to(dev)  # width, height
    im_w = im_sizes[:, 0]
    im_h = im_sizes[:, 1]

    bbox_stat = bboxes[:, 1:].clone()
    bbox_stat[:, :, 2] *= 200
    num_frames = bbox_stat.shape[0]
    bboxes_stat = fetch_bbox_stat(bbox_stat, im_w[1:], im_h[1:])
    bboxes_ego = (
        torch.FloatTensor(np.array([1, 1, 2800, 2000]))[None, None, :]
        .repeat(num_frames, 1, 1)
        .to(dev)
    )
    bboxes_all = torch.cat((bboxes_ego, bboxes_stat), dim=1)

    hand_valid_r = bbox_jts_to_valid(bboxes_all.clone(), j2d_r)
    hand_valid_l = bbox_jts_to_valid(bboxes_all.clone(), j2d_l)
    is_valid = bbox_jts_to_valid(bboxes_all.clone(), v2d_o)  # center of object verts

    is_valid = is_valid[:, :, 0]

    # right_valid if at least 3 joints are valid and root is inside the bbox
    # is_valid if center of object is inside bbox
    right_valid = hand_valid_r[:, :, 0] * (hand_valid_r.sum(dim=2) >= 3).long()
    left_valid = hand_valid_l[:, :, 0] * (hand_valid_l.sum(dim=2) >= 3).long()
    out = {"is_valid": is_valid, "left_valid": left_valid, "right_valid": right_valid}
    return out


def fetch_bbox_stat(bbox_stat, im_w, im_h):
    assert isinstance(bbox_stat, torch.Tensor)
    assert isinstance(im_w, torch.Tensor)
    assert isinstance(im_h, torch.Tensor)
    num_frames, num_views = bbox_stat.shape[:2]
    assert im_w.shape[0] == num_views
    assert im_h.shape == im_w.shape
    cx = bbox_stat[:, :, 0]
    cy = bbox_stat[:, :, 1]
    scale = bbox_stat[:, :, 2]

    # bbox to define limits of xy
    xmin = torch.clamp(cx - scale / 2, 1)
    ymin = torch.clamp(cy - scale / 2, 1)

    im_w_batch = im_w[None, :].repeat(num_frames, 1)
    im_h_batch = im_h[None, :].repeat(num_frames, 1)

    xmax = torch.minimum(cx + scale / 2, im_w_batch)
    ymax = torch.minimum(cy + scale / 2, im_h_batch)
    boxes = torch.stack((xmin, ymin, xmax, ymax), dim=2)
    return boxes


def process_seq(task, export_verts=False):
    """
    Process one sequence
    """

    with torch.no_grad():
        mano_p, dev, statcams, layers, pbar = task

        image_sizes = {}
        for sub in misc.keys():
            image_sizes[sub] = misc[sub]["image_size"]
        sub = mano_p.split("/")[-2]

        curr_batch_size = None

        cams = statcams[sub]
        world2cam = cams["world2cam"].to(dev)
        intris_mat = cams["intris_mat"].to(dev)

        loader = construct_loader(mano_p)
        out_world_list = []
        out_views_list = []
        out_2d_list = []
        out_bbox_list = []
        batch_list = []
        for batch in loader:
            batch = thing.thing2dev(batch, dev)
            batch_size = batch["rot_r"].shape[0]
            if batch_size != curr_batch_size:
                curr_batch_size = batch_size
                smplx_m = human_models.build_subject_smplx(curr_batch_size, sub).to(dev)
            out_world, out_views, out_all_2d, bbox = process_batch(
                batch,
                layers,
                smplx_m,
                world2cam,
                intris_mat,
                image_sizes,
                sub,
                export_verts,
            )
            out_world_list.append(out_world)
            out_views_list.append(out_views)
            out_2d_list.append(out_all_2d)
            out_bbox_list.append(bbox)

            batch.pop("query_names")
            batch_list.append(batch)

        out_world_list = ld_utils.ld2dl(out_world_list)
        out_views_list = ld_utils.ld2dl(out_views_list)
        out_2d_list = ld_utils.ld2dl(out_2d_list)
        batch_list = ld_utils.ld2dl(batch_list)
        out_bbox_list = torch.cat(out_bbox_list, dim=0)

        out_world_list = cat_dl(out_world_list, dim=0)
        out_views_list = cat_dl(out_views_list, dim=0)
        out_2d_list = cat_dl(out_2d_list, dim=0)
        batch_list = cat_dl(batch_list, dim=0)

        out_world_list = thing.thing2np(out_world_list)
        out_views_list = thing.thing2np(out_views_list)
        out_2d_list = thing.thing2np(out_2d_list)
        batch_list = thing.thing2np(batch_list)
        out_bbox_list = out_bbox_list.cpu().detach().numpy()

        out = {}
        out["world_coord"] = out_world_list
        out["cam_coord"] = out_views_list
        out["2d"] = out_2d_list
        out["bbox"] = out_bbox_list
        out["params"] = batch_list

        sid, seqname = mano_p.split("/")[-2:]
        if export_verts:
            out_p = f"./outputs/processed_verts/seqs/{sid}/{seqname}"
        else:
            out_p = f"./outputs/processed/seqs/{sid}/{seqname}"
        out_p = out_p.replace(".mano", "").replace("/annot", "")
        out_folder = op.dirname(out_p)

        if not op.exists(out_folder):
            os.makedirs(out_folder)

        pbar.set_description(f"Save to {out_p}")
        np.save(out_p, out)
