import os.path as op

import numpy as np
import torch
import trimesh

import common.viewer as viewer_utils
from common.body_models import build_layers, seal_mano_mesh
from common.xdict import xdict
from src.extraction.interface import prepare_data
from src.extraction.keys.vis_pose import KEYS as keys


def construct_meshes(exp_folder, seq_name, flag, side_angle=None, zoom_out=0.5):
    exp_key = exp_folder.split("/")[1]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    layers = build_layers(device)

    data = prepare_data(
        seq_name,
        exp_key,
        keys,
        layers,
        device,
        task="pose",
        eval_p=op.join(exp_folder, "eval"),
    )

    # load object faces
    obj_name = seq_name.split("_")[1]
    f3d_o = trimesh.load(
        f"./data/arctic_data/data/meta/object_vtemplates/{obj_name}/mesh.obj",
        process=False,
    ).faces

    # center verts
    v3d_r = data[f"{flag}.mano.v3d.cam.r"]
    v3d_l = data[f"{flag}.mano.v3d.cam.l"]
    v3d_o = data[f"{flag}.object.v.cam"]
    cam_t = data[f"{flag}.object.cam_t"]
    v3d_r -= cam_t[:, None, :]
    v3d_l -= cam_t[:, None, :]
    v3d_o -= cam_t[:, None, :]

    # seal MANO mesh
    f3d_r = torch.LongTensor(layers["right"].faces.astype(np.int64))
    f3d_l = torch.LongTensor(layers["left"].faces.astype(np.int64))
    v3d_r, f3d_r = seal_mano_mesh(v3d_r, f3d_r, True)
    v3d_l, f3d_l = seal_mano_mesh(v3d_l, f3d_l, False)

    # AIT meshes
    hand_color = "white"
    object_color = "light-blue"
    right = {
        "v3d": v3d_r.numpy(),
        "f3d": f3d_r.numpy(),
        "vc": None,
        "name": "right",
        "color": hand_color,
    }
    left = {
        "v3d": v3d_l.numpy(),
        "f3d": f3d_l.numpy(),
        "vc": None,
        "name": "left",
        "color": hand_color,
    }
    obj = {
        "v3d": v3d_o.numpy(),
        "f3d": f3d_o,
        "vc": None,
        "name": "object",
        "color": object_color,
    }

    meshes = viewer_utils.construct_viewer_meshes(
        {
            "right": right,
            "left": left,
            "object": obj,
        },
        draw_edges=False,
        flat_shading=True,
    )
    data = xdict(data).to_np()
    return meshes, data
