import numpy as np
import torch
from PIL import Image

import common.viewer as viewer_utils
from common.mesh import Mesh
from common.viewer import ViewerData


def construct_hand_meshes(cam_data, layers, view_idx, distort):
    if view_idx == 0 and distort:
        view_idx = 9
    v3d_r = cam_data["verts.right"][:, view_idx]
    v3d_l = cam_data["verts.left"][:, view_idx]

    right = {
        "v3d": v3d_r,
        "f3d": layers["right"].faces,
        "vc": None,
        "name": "right",
        "color": "white",
    }
    left = {
        "v3d": v3d_l,
        "f3d": layers["left"].faces,
        "vc": None,
        "name": "left",
        "color": "white",
    }
    return right, left


def construct_object_meshes(cam_data, obj_name, layers, view_idx, distort):
    if view_idx == 0 and distort:
        view_idx = 9
    v3d_o = cam_data["verts.object"][:, view_idx]
    f3d_o = Mesh(
        filename=f"./data/arctic_data/data/meta/object_vtemplates/{obj_name}/mesh.obj"
    ).faces

    obj = {
        "v3d": v3d_o,
        "f3d": f3d_o,
        "vc": None,
        "name": "object",
        "color": "light-blue",
    }
    return obj


def construct_smplx_meshes(cam_data, layers, view_idx, distort):
    assert not distort, "Distortion rendering not supported for SMPL-X"
    # We use the following algorithm to render meshes with distortion effects:
    # VR Distortion Correction Using Vertex Displacement
    # https://stackoverflow.com/questions/44489686/camera-lens-distortion-in-opengl
    # However, this method creates artifacts when vertices are too close to the camera.

    if view_idx == 0 and distort:
        view_idx = 9

    v3d_s = cam_data["verts.smplx"][:, view_idx]

    smplx_mesh = {
        "v3d": v3d_s,
        "f3d": layers["smplx"].faces,
        "vc": None,
        "name": "smplx",
        "color": "rice",
    }

    return smplx_mesh


def construct_meshes(
    seq_p,
    layers,
    use_mano,
    use_object,
    use_smplx,
    no_image,
    use_distort,
    view_idx,
    subject_meta,
):
    # load
    data = np.load(seq_p, allow_pickle=True).item()
    cam_data = data["cam_coord"]
    data_params = data["params"]
    # unpack
    subject = seq_p.split("/")[-2]
    seq_name = seq_p.split("/")[-1].split(".")[0]
    obj_name = seq_name.split("_")[0]

    num_frames = cam_data["verts.right"].shape[0]

    # camera intrinsics
    if view_idx == 0:
        K = torch.FloatTensor(data_params["K_ego"][0].copy())
    else:
        K = torch.FloatTensor(
            np.array(subject_meta[subject]["intris_mat"][view_idx - 1])
        )

    # image names
    vidx = np.arange(num_frames)
    image_idx = vidx + subject_meta[subject]["ioi_offset"]
    imgnames = [
        f"./data/arctic_data/data/images/{subject}/{seq_name}/{view_idx}/{idx:05d}.jpg"
        for idx in image_idx
    ]

    # construct meshes
    vis_dict = {}
    if use_mano:
        right, left = construct_hand_meshes(cam_data, layers, view_idx, use_distort)
        vis_dict["right"] = right
        vis_dict["left"] = left
    if use_smplx:
        smplx_mesh = construct_smplx_meshes(cam_data, layers, view_idx, use_distort)
        vis_dict["smplx"] = smplx_mesh
    if use_object:
        obj = construct_object_meshes(cam_data, obj_name, layers, view_idx, use_distort)
        vis_dict["object"] = obj

    meshes = viewer_utils.construct_viewer_meshes(
        vis_dict, draw_edges=False, flat_shading=False
    )

    num_frames = len(imgnames)
    Rt = np.zeros((num_frames, 3, 4))
    Rt[:, :3, :3] = np.eye(3)
    Rt[:, 1:3, :3] *= -1.0

    im = Image.open(imgnames[0])
    cols, rows = im.size
    if no_image:
        imgnames = None

    data = ViewerData(Rt, K, cols, rows, imgnames)
    return meshes, data
