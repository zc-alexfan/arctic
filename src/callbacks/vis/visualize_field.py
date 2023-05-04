import copy

import matplotlib
import numpy as np
import torch

import common.thing as thing
import common.vis_utils as vis_utils
from common.data_utils import denormalize_images
from common.mesh import Mesh
from common.torch_utils import unpad_vtensor

cmap = matplotlib.cm.get_cmap("plasma")


def dist2vc_hands_cnt(contact_r):
    contact_r = torch.clamp(contact_r.clone(), 0, 0.1) / 0.1
    contact_r = 1 - contact_r
    vc = cmap(contact_r)
    return vc


def dist2vc_hands(contact_r, decision_bnd):
    return dist2vc_hands_cnt(contact_r)


def dist2vc_obj(contact_t, norm_f):
    return dist2vc_hands(contact_t, norm_f)


def render_result(
    renderer,
    vertices_r,
    vertices_l,
    vc_r,
    vc_l,
    mano_faces_r,
    mano_faces_l,
    vertices_t,
    vc_t,
    faces_t,
    r_valid,
    l_valid,
    K,
    img,
):
    img = img.permute(1, 2, 0).cpu().numpy()
    mesh_top = Mesh(
        v=thing.thing2np(vertices_t),
        f=thing.thing2np(faces_t),
        vc=vc_t,
    )

    # render only valid meshes
    meshes = []
    mesh_names = []
    if r_valid:
        mesh_r = Mesh(
            v=vertices_r,
            f=mano_faces_r,
            vc=vc_r,
        )
        meshes.append(mesh_r)
        mesh_names.append("right")

    if l_valid:
        mesh_l = Mesh(v=vertices_l, f=mano_faces_l, vc=vc_l)

        meshes.append(mesh_l)
        mesh_names.append("left")

    meshes = meshes + [mesh_top]
    mesh_names = mesh_names + ["object"]

    # render meshes
    render_img_img = render_meshes(
        renderer, meshes, mesh_names, K, img, sideview_angle=None
    )

    # render in different angles
    render_img_angles = []
    for angle in list(np.linspace(45, 300, 3)):
        render_img_angle = render_meshes(
            renderer, meshes, mesh_names, K, img=None, sideview_angle=angle
        )
        render_img_angles.append(render_img_angle)
    render_img_angles = [render_img_img] + render_img_angles
    render_img = np.concatenate(render_img_angles, axis=0)
    return render_img


def render_meshes(renderer, meshes, mesh_names, K, img, sideview_angle):
    materials = None
    rend_img = renderer.render_meshes_pose(
        cam_transl=None,
        meshes=meshes,
        image=img,
        materials=materials,
        sideview_angle=sideview_angle,
        K=K,
    )
    return rend_img


def visualize_all(_vis_dict, max_examples, renderer, postfix, no_tqdm):
    # unpack
    vis_dict = copy.deepcopy(_vis_dict)
    K = vis_dict["meta_info.intrinsics"]
    image_ids = [
        "/".join(key.split("/")[-5:]).replace(".jpg", "")
        for key in vis_dict["meta_info.imgname"]
    ]
    images = denormalize_images(vis_dict["inputs.img"])
    vis_dict.pop("inputs.img", None)
    vis_dict["vis.images"] = images
    vis_dict["vis.image_ids"] = image_ids

    # unpack MANO
    pred_vertices_r_cam = vis_dict["targets.mano.v3d.cam.r"]
    pred_vertices_l_cam = vis_dict["targets.mano.v3d.cam.l"]
    gt_vertices_r_cam = vis_dict["targets.mano.v3d.cam.r"]
    gt_vertices_l_cam = vis_dict["targets.mano.v3d.cam.l"]
    mano_faces_r = vis_dict["meta_info.mano.faces.r"]
    mano_faces_l = vis_dict["meta_info.mano.faces.l"]

    # unpack object
    gt_obj_vtop_cam = unpad_vtensor(
        vis_dict["targets.object.v.cam"], vis_dict["targets.object.v_len"]
    )
    gt_obj_ftop = unpad_vtensor(
        vis_dict["targets.object.f"], vis_dict["targets.object.f_len"]
    )
    pred_obj_vtop_cam = unpad_vtensor(
        vis_dict["targets.object.v.cam"],
        vis_dict["targets.object.v_len"],
    )

    # valid flag
    right_valid = vis_dict["targets.right_valid"].bool()
    left_valid = vis_dict["targets.left_valid"].bool()

    # unpack dist
    gt_dist_r = vis_dict["targets.dist.ro"]
    gt_dist_l = vis_dict["targets.dist.lo"]
    gt_dist_or = vis_dict["targets.dist.or"]
    gt_dist_ol = vis_dict["targets.dist.ol"]

    pred_dist_r = vis_dict["pred.dist.ro"]
    pred_dist_l = vis_dict["pred.dist.lo"]
    pred_dist_or = vis_dict["pred.dist.or"]
    pred_dist_ol = vis_dict["pred.dist.ol"]

    im_list = []
    # rendering
    for idx in range(min(len(image_ids), max_examples)):
        r_valid = right_valid[idx]
        l_valid = left_valid[idx]
        K_i = K[idx]
        image_id = image_ids[idx]

        top_len = vis_dict["targets.object.v_len"][idx]

        # dist to vertex color
        max_dist = 0.10
        gt_vc_r = dist2vc_hands(gt_dist_r[idx], max_dist)
        gt_vc_l = dist2vc_hands(gt_dist_l[idx], max_dist)
        gt_vc_or = dist2vc_obj(gt_dist_or[idx][:top_len], max_dist)
        gt_vc_ol = dist2vc_obj(gt_dist_ol[idx][:top_len], max_dist)

        pred_vc_r = dist2vc_hands(pred_dist_r[idx], max_dist)
        pred_vc_l = dist2vc_hands(pred_dist_l[idx], max_dist)
        pred_vc_or = dist2vc_obj(pred_dist_or[idx][:top_len], max_dist)
        pred_vc_ol = dist2vc_obj(pred_dist_ol[idx][:top_len], max_dist)

        # render GT
        image_list = []
        image_list.append(images[idx].permute(1, 2, 0).cpu().numpy())
        # render one hand at a time
        image_gt_r = render_result(
            renderer,
            gt_vertices_r_cam[idx],
            gt_vertices_l_cam[idx],
            gt_vc_r,
            gt_vc_l,
            mano_faces_r,
            mano_faces_l,
            gt_obj_vtop_cam[idx],
            gt_vc_or,
            gt_obj_ftop[idx],
            r_valid,
            False,
            K_i,
            images[idx],
        )
        image_gt_l = render_result(
            renderer,
            gt_vertices_r_cam[idx],
            gt_vertices_l_cam[idx],
            gt_vc_r,
            gt_vc_l,
            mano_faces_r,
            mano_faces_l,
            gt_obj_vtop_cam[idx],
            gt_vc_ol,
            gt_obj_ftop[idx],
            False,
            l_valid,
            K_i,
            images[idx],
        )

        # prediction
        image_pred_r = render_result(
            renderer,
            pred_vertices_r_cam[idx],
            pred_vertices_l_cam[idx],
            pred_vc_r,
            pred_vc_l,
            mano_faces_r,
            mano_faces_l,
            pred_obj_vtop_cam[idx],
            pred_vc_or,
            gt_obj_ftop[idx],
            r_valid,
            False,
            K_i,
            images[idx],
        )
        image_pred_l = render_result(
            renderer,
            pred_vertices_r_cam[idx],
            pred_vertices_l_cam[idx],
            pred_vc_r,
            pred_vc_l,
            mano_faces_r,
            mano_faces_l,
            pred_obj_vtop_cam[idx],
            pred_vc_ol,
            gt_obj_ftop[idx],
            False,
            l_valid,
            K_i,
            images[idx],
        )

        image_list.append(image_pred_r)
        image_list.append(image_gt_r)

        image_list.append(image_pred_l)
        image_list.append(image_gt_l)

        image_pred = vis_utils.im_list_to_plt(
            image_list,
            figsize=(15, 8),
            title_list=[
                "input image",
                "PRED-R",
                "GT-R",
                "PRED-L",
                "GT-L",
            ],
        )
        im_list.append({"fig_name": f"{image_id}", "im": image_pred})

    # post fix image list
    im_list_postfix = []
    for im in im_list:
        im["fig_name"] += postfix
        im_list_postfix.append(im)
    return im_list
