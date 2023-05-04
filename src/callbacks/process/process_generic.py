import torch

import src.utils.interfield as inter


def prepare_mano_template(batch_size, mano_layer, mesh_sampler, is_right):
    root_idx = 0

    # Generate T-pose template mesh
    template_pose = torch.zeros((1, 48))
    template_pose = template_pose.cuda()
    template_betas = torch.zeros((1, 10)).cuda()
    out = mano_layer(
        betas=template_betas,
        hand_pose=template_pose[:, 3:],
        global_orient=template_pose[:, :3],
        transl=None,
    )
    template_3d_joints = out.joints
    template_vertices = out.vertices
    template_vertices_sub = mesh_sampler.downsample(template_vertices, is_right)

    # normalize
    template_root = template_3d_joints[:, root_idx, :]
    template_3d_joints = template_3d_joints - template_root[:, None, :]
    template_vertices = template_vertices - template_root[:, None, :]
    template_vertices_sub = template_vertices_sub - template_root[:, None, :]

    # concatinate template joints and template vertices, and then duplicate to batch size
    ref_vertices = torch.cat([template_3d_joints, template_vertices_sub], dim=1)
    ref_vertices = ref_vertices.expand(batch_size, -1, -1)

    ref_vertices_full = torch.cat([template_3d_joints, template_vertices], dim=1)
    ref_vertices_full = ref_vertices_full.expand(batch_size, -1, -1)
    return ref_vertices, ref_vertices_full


def prepare_templates(
    batch_size,
    mano_r,
    mano_l,
    mesh_sampler,
    arti_head,
    query_names,
):
    v0_r, v0_r_full = prepare_mano_template(
        batch_size, mano_r, mesh_sampler, is_right=True
    )
    v0_l, v0_l_full = prepare_mano_template(
        batch_size, mano_l, mesh_sampler, is_right=False
    )
    (v0_o, pidx, v0_full, mask) = prepare_object_template(
        batch_size,
        arti_head.object_tensors,
        query_names,
    )
    CAM_R, CAM_L, CAM_O = list(range(100))[-3:]
    cams = (
        torch.FloatTensor([CAM_R, CAM_L, CAM_O]).view(1, 3, 1).repeat(batch_size, 1, 3)
        / 100
    )
    cams = cams.to(v0_r.device)
    return (
        v0_r,
        v0_l,
        v0_o,
        pidx,
        v0_r_full,
        v0_l_full,
        v0_full,
        mask,
        cams,
    )


def prepare_object_template(batch_size, object_tensors, query_names):
    template_angles = torch.zeros((batch_size, 1)).cuda()
    template_rot = torch.zeros((batch_size, 3)).cuda()
    out = object_tensors.forward(
        angles=template_angles,
        global_orient=template_rot,
        transl=None,
        query_names=query_names,
    )
    ref_vertices = out["v_sub"]
    parts_idx = out["parts_ids"]

    mask = out["mask"]

    ref_mean = ref_vertices.mean(dim=1)[:, None, :]
    ref_vertices -= ref_mean

    v_template = out["v"]
    return (ref_vertices, parts_idx, v_template, mask)


def prepare_interfield(targets, max_dist):
    dist_min = 0.0
    dist_max = max_dist
    dist_ro, dist_ro_idx = inter.compute_dist_mano_to_obj(
        targets["mano.v3d.cam.r"],
        targets["object.v.cam"],
        targets["object.v_len"],
        dist_min,
        dist_max,
    )
    dist_lo, dist_lo_idx = inter.compute_dist_mano_to_obj(
        targets["mano.v3d.cam.l"],
        targets["object.v.cam"],
        targets["object.v_len"],
        dist_min,
        dist_max,
    )
    dist_or, dist_or_idx = inter.compute_dist_obj_to_mano(
        targets["mano.v3d.cam.r"],
        targets["object.v.cam"],
        targets["object.v_len"],
        dist_min,
        dist_max,
    )
    dist_ol, dist_ol_idx = inter.compute_dist_obj_to_mano(
        targets["mano.v3d.cam.l"],
        targets["object.v.cam"],
        targets["object.v_len"],
        dist_min,
        dist_max,
    )

    targets["dist.ro"] = dist_ro
    targets["dist.lo"] = dist_lo
    targets["dist.or"] = dist_or
    targets["dist.ol"] = dist_ol

    targets["idx.ro"] = dist_ro_idx
    targets["idx.lo"] = dist_lo_idx
    targets["idx.or"] = dist_or_idx
    targets["idx.ol"] = dist_ol_idx
    return targets
