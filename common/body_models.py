import json

import numpy as np
import torch
from smplx import MANO

from common.mesh import Mesh


class MANODecimator:
    def __init__(self):
        data = np.load(
            "./data/arctic_data/data/meta/mano_decimator_195.npy", allow_pickle=True
        ).item()
        mydata = {}
        for key, val in data.items():
            # only consider decimation matrix so far
            if "D" in key:
                mydata[key] = torch.FloatTensor(val)
        self.data = mydata

    def downsample(self, verts, is_right):
        dev = verts.device
        flag = "right" if is_right else "left"
        if self.data[f"D_{flag}"].device != dev:
            self.data[f"D_{flag}"] = self.data[f"D_{flag}"].to(dev)
        D = self.data[f"D_{flag}"]
        batch_size = verts.shape[0]
        D_batch = D[None, :, :].repeat(batch_size, 1, 1)
        verts_sub = torch.bmm(D_batch, verts)
        return verts_sub


MODEL_DIR = "./data/body_models/mano"

SEAL_FACES_R = [
    [120, 108, 778],
    [108, 79, 778],
    [79, 78, 778],
    [78, 121, 778],
    [121, 214, 778],
    [214, 215, 778],
    [215, 279, 778],
    [279, 239, 778],
    [239, 234, 778],
    [234, 92, 778],
    [92, 38, 778],
    [38, 122, 778],
    [122, 118, 778],
    [118, 117, 778],
    [117, 119, 778],
    [119, 120, 778],
]

# vertex ids around the ring of the wrist
CIRCLE_V_ID = np.array(
    [108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120],
    dtype=np.int64,
)


def seal_mano_mesh(v3d, faces, is_rhand):
    # v3d: B, 778, 3
    # faces: 1538, 3
    # output: v3d(B, 779, 3); faces (1554, 3)

    seal_faces = torch.LongTensor(np.array(SEAL_FACES_R)).to(faces.device)
    if not is_rhand:
        # left hand
        seal_faces = seal_faces[:, np.array([1, 0, 2])]  # invert face normal
    centers = v3d[:, CIRCLE_V_ID].mean(dim=1)[:, None, :]
    sealed_vertices = torch.cat((v3d, centers), dim=1)
    faces = torch.cat((faces, seal_faces), dim=0)
    return sealed_vertices, faces


def build_layers(device=None):
    from common.object_tensors import ObjectTensors

    layers = {
        "right": build_mano_aa(True),
        "left": build_mano_aa(False),
        "object_tensors": ObjectTensors(),
    }

    if device is not None:
        layers["right"] = layers["right"].to(device)
        layers["left"] = layers["left"].to(device)
        layers["object_tensors"].to(device)
    return layers


MANO_MODEL_DIR = "./data/body_models/mano"
SMPLX_MODEL_P = {
    "male": "./data/body_models/smplx/SMPLX_MALE.npz",
    "female": "./data/body_models/smplx/SMPLX_FEMALE.npz",
    "neutral": "./data/body_models/smplx/SMPLX_NEUTRAL.npz",
}


def build_smplx(batch_size, gender, vtemplate):
    import smplx

    subj_m = smplx.create(
        model_path=SMPLX_MODEL_P[gender],
        model_type="smplx",
        gender=gender,
        num_pca_comps=45,
        v_template=vtemplate,
        flat_hand_mean=True,
        use_pca=False,
        batch_size=batch_size,
        # batch_size=320,
    )
    return subj_m


def build_subject_smplx(batch_size, subject_id):
    with open("./data/arctic_data/data/meta/misc.json", "r") as f:
        misc = json.load(f)
    vtemplate_p = f"./data/arctic_data/data/meta/subject_vtemplates/{subject_id}.obj"
    mesh = Mesh(filename=vtemplate_p)
    vtemplate = mesh.v
    gender = misc[subject_id]["gender"]
    return build_smplx(batch_size, gender, vtemplate)


def build_mano_aa(is_rhand, create_transl=False, flat_hand=False):
    return MANO(
        MODEL_DIR,
        create_transl=create_transl,
        use_pca=False,
        flat_hand_mean=flat_hand,
        is_rhand=is_rhand,
    )


def construct_layers(dev):
    mano_layers = {
        "right": build_mano_aa(True, create_transl=True, flat_hand=False),
        "left": build_mano_aa(False, create_transl=True, flat_hand=False),
        "smplx": build_smplx(1, "neutral", None),
    }
    for layer in mano_layers.values():
        layer.to(dev)
    return mano_layers
