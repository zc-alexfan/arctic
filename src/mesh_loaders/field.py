import os.path as op

import matplotlib
import numpy as np
import torch

import common.viewer as viewer_utils
from common.body_models import build_layers, seal_mano_mesh
from common.mesh import Mesh
from common.xdict import xdict
from src.extraction.interface import prepare_data
from src.extraction.keys.vis_field import KEYS as keys


def dist2vc(dist_r, dist_l, dist_o, ccmap):
    vc_r, vc_l, vc_o = viewer_utils.dist2vc(dist_r, dist_l, dist_o, ccmap)

    vc_r_pad = np.zeros((vc_r.shape[0], vc_r.shape[1] + 1, 4))
    vc_l_pad = np.zeros((vc_l.shape[0], vc_l.shape[1] + 1, 4))

    # sealed vertex to pre-defined color
    vc_r_pad[:, -1, 0] = 0.4
    vc_l_pad[:, -1, 0] = 0.4
    vc_r_pad[:, -1, 1] = 0.2
    vc_l_pad[:, -1, 1] = 0.2
    vc_r_pad[:, -1, 2] = 0.3
    vc_l_pad[:, -1, 2] = 0.3
    vc_r_pad[:, -1, 3] = 1.0
    vc_l_pad[:, -1, 3] = 1.0
    vc_r_pad[:, :-1, :] = vc_r
    vc_l_pad[:, :-1, :] = vc_l

    vc_r = vc_r_pad
    vc_l = vc_l_pad
    return vc_r, vc_l, vc_o


def construct_meshes(exp_folder, seq_name, flag, mode, side_angle=None, zoom_out=0.5):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    layers = build_layers(device)

    exp_key = exp_folder.split("/")[1]
    # load data
    data = prepare_data(
        seq_name,
        exp_key,
        keys,
        layers,
        device,
        task="field",
        eval_p=op.join(exp_folder, "eval"),
    )

    # load object faces
    obj_name = seq_name.split("_")[1]
    f3d_o = Mesh(
        filename=f"./data/arctic_data/data/meta/object_vtemplates/{obj_name}/mesh.obj"
    ).f

    # only show predicted dist < 0.1
    num_frames = data["targets.dist.or"].shape[0]
    num_verts = data["targets.dist.or"].shape[1]
    data["pred.dist.or"][:num_frames, :num_verts][data["targets.dist.or"] == 0.1] = 0.1
    data["pred.dist.ol"][:num_frames, :num_verts][data["targets.dist.ol"] == 0.1] = 0.1
    data["pred.dist.ro"][:num_frames, :num_verts][data["targets.dist.ro"] == 0.1] = 0.1
    data["pred.dist.lo"][:num_frames, :num_verts][data["targets.dist.lo"] == 0.1] = 0.1

    # center verts
    v3d_r = data[f"targets.mano.v3d.cam.r"]
    v3d_l = data[f"targets.mano.v3d.cam.l"]
    v3d_o = data[f"targets.object.v.cam"]
    cam_t = data[f"targets.object.cam_t"]
    v3d_r -= cam_t[:, None, :]
    v3d_l -= cam_t[:, None, :]
    v3d_o -= cam_t[:, None, :]

    # seal MANO meshes
    f3d_r = torch.LongTensor(layers["right"].faces.astype(np.int64))
    f3d_l = torch.LongTensor(layers["left"].faces.astype(np.int64))
    v3d_r, f3d_r = seal_mano_mesh(v3d_r, f3d_r, True)
    v3d_l, f3d_l = seal_mano_mesh(v3d_l, f3d_l, False)

    if "_l" in mode:
        mydist_o = data[f"{flag}.dist.ol"]
    else:
        mydist_o = data[f"{flag}.dist.or"]

    ccmap = matplotlib.cm.get_cmap("plasma")
    vc_r, vc_l, vc_o = dist2vc(
        data[f"{flag}.dist.ro"], data[f"{flag}.dist.lo"], mydist_o, ccmap
    )

    right = {
        "v3d": v3d_r.numpy(),
        "f3d": f3d_r.numpy(),
        "vc": vc_r,
        "name": "right",
        "color": "none",
    }
    left = {
        "v3d": v3d_l.numpy(),
        "f3d": f3d_l.numpy(),
        "vc": vc_l,
        "name": "left",
        "color": "none",
    }
    obj = {
        "v3d": v3d_o.numpy(),
        "f3d": f3d_o,
        "vc": vc_o,
        "name": "object",
        "color": "none",
    }
    meshes = viewer_utils.construct_viewer_meshes(
        {"right": right, "left": left, "object": obj},
        draw_edges=False,
        flat_shading=True,
    )
    data = xdict(data).to_np()

    # pred_field uses GT cam_t for vis
    data["pred.object.cam_t"] = data["targets.object.cam_t"]
    return meshes, data
