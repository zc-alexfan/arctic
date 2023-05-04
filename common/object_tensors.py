import json
import os.path as op
import sys

import numpy as np
import torch
import torch.nn as nn
import trimesh
from easydict import EasyDict
from scipy.spatial.distance import cdist

sys.path = [".."] + sys.path
import common.thing as thing
from common.rot import axis_angle_to_quaternion, quaternion_apply
from common.torch_utils import pad_tensor_list
from common.xdict import xdict

# objects to consider for training so far
OBJECTS = [
    "capsulemachine",
    "box",
    "ketchup",
    "laptop",
    "microwave",
    "mixer",
    "notebook",
    "espressomachine",
    "waffleiron",
    "scissors",
    "phone",
]


class ObjectTensors(nn.Module):
    def __init__(self):
        super(ObjectTensors, self).__init__()
        self.obj_tensors = thing.thing2dev(construct_obj_tensors(OBJECTS), "cpu")
        self.dev = None

    def forward_7d_batch(
        self,
        angles: (None, torch.Tensor),
        global_orient: (None, torch.Tensor),
        transl: (None, torch.Tensor),
        query_names: list,
        fwd_template: bool,
    ):
        self._sanity_check(angles, global_orient, transl, query_names, fwd_template)

        # store output
        out = xdict()

        # meta info
        obj_idx = np.array(
            [self.obj_tensors["names"].index(name) for name in query_names]
        )
        out["diameter"] = self.obj_tensors["diameter"][obj_idx]
        out["f"] = self.obj_tensors["f"][obj_idx]
        out["f_len"] = self.obj_tensors["f_len"][obj_idx]
        out["v_len"] = self.obj_tensors["v_len"][obj_idx]

        max_len = out["v_len"].max()
        out["v"] = self.obj_tensors["v"][obj_idx][:, :max_len]
        out["mask"] = self.obj_tensors["mask"][obj_idx][:, :max_len]
        out["v_sub"] = self.obj_tensors["v_sub"][obj_idx]
        out["parts_ids"] = self.obj_tensors["parts_ids"][obj_idx][:, :max_len]
        out["parts_sub_ids"] = self.obj_tensors["parts_sub_ids"][obj_idx]

        if fwd_template:
            return out

        # articulation + global rotation
        quat_arti = axis_angle_to_quaternion(self.obj_tensors["z_axis"] * angles)
        quat_global = axis_angle_to_quaternion(global_orient.view(-1, 3))

        # mm
        # collect entities to be transformed
        tf_dict = xdict()
        tf_dict["v_top"] = out["v"].clone()
        tf_dict["v_sub_top"] = out["v_sub"].clone()
        tf_dict["v_bottom"] = out["v"].clone()
        tf_dict["v_sub_bottom"] = out["v_sub"].clone()
        tf_dict["bbox_top"] = self.obj_tensors["bbox_top"][obj_idx]
        tf_dict["bbox_bottom"] = self.obj_tensors["bbox_bottom"][obj_idx]
        tf_dict["kp_top"] = self.obj_tensors["kp_top"][obj_idx]
        tf_dict["kp_bottom"] = self.obj_tensors["kp_bottom"][obj_idx]

        # articulate top parts
        for key, val in tf_dict.items():
            if "top" in key:
                val_rot = quaternion_apply(quat_arti[:, None, :], val)
                tf_dict.overwrite(key, val_rot)

        # global rotation for all
        for key, val in tf_dict.items():
            val_rot = quaternion_apply(quat_global[:, None, :], val)
            if transl is not None:
                val_rot = val_rot + transl[:, None, :]
            tf_dict.overwrite(key, val_rot)

        # prep output
        top_idx = out["parts_ids"] == 1
        v_tensor = tf_dict["v_bottom"].clone()
        v_tensor[top_idx, :] = tf_dict["v_top"][top_idx, :]

        top_idx = out["parts_sub_ids"] == 1
        v_sub_tensor = tf_dict["v_sub_bottom"].clone()
        v_sub_tensor[top_idx, :] = tf_dict["v_sub_top"][top_idx, :]

        bbox = torch.cat((tf_dict["bbox_top"], tf_dict["bbox_bottom"]), dim=1)
        kp3d = torch.cat((tf_dict["kp_top"], tf_dict["kp_bottom"]), dim=1)

        out.overwrite("v", v_tensor)
        out.overwrite("v_sub", v_sub_tensor)
        out.overwrite("bbox3d", bbox)
        out.overwrite("kp3d", kp3d)
        return out

    def forward(self, angles, global_orient, transl, query_names):
        out = self.forward_7d_batch(
            angles, global_orient, transl, query_names, fwd_template=False
        )
        return out

    def forward_template(self, query_names):
        out = self.forward_7d_batch(
            angles=None,
            global_orient=None,
            transl=None,
            query_names=query_names,
            fwd_template=True,
        )
        return out

    def to(self, dev):
        self.obj_tensors = thing.thing2dev(self.obj_tensors, dev)
        self.dev = dev

    def _sanity_check(self, angles, global_orient, transl, query_names, fwd_template):
        # sanity check
        if not fwd_template:
            # assume transl is in meter
            if transl is not None:
                transl = transl * 1000  # mm

            batch_size = angles.shape[0]
            assert angles.shape == (batch_size, 1)
            assert global_orient.shape == (batch_size, 3)
            if transl is not None:
                assert isinstance(transl, torch.Tensor)
                assert transl.shape == (batch_size, 3)
            assert len(query_names) == batch_size


def construct_obj(object_model_p):
    # load vtemplate
    mesh_p = op.join(object_model_p, "mesh.obj")
    parts_p = op.join(object_model_p, f"parts.json")
    json_p = op.join(object_model_p, "object_params.json")
    obj_name = op.basename(object_model_p)

    top_sub_p = f"./data/arctic_data/data/meta/object_vtemplates/{obj_name}/top_keypoints_300.json"
    bottom_sub_p = top_sub_p.replace("top_", "bottom_")
    with open(top_sub_p, "r") as f:
        sub_top = np.array(json.load(f)["keypoints"])

    with open(bottom_sub_p, "r") as f:
        sub_bottom = np.array(json.load(f)["keypoints"])
    sub_v = np.concatenate((sub_top, sub_bottom), axis=0)

    with open(parts_p, "r") as f:
        parts = np.array(json.load(f), dtype=np.bool)

    assert op.exists(mesh_p), f"Not found: {mesh_p}"

    mesh = trimesh.exchange.load.load_mesh(mesh_p, process=False)
    mesh_v = mesh.vertices

    mesh_f = torch.LongTensor(mesh.faces)
    vidx = np.argmin(cdist(sub_v, mesh_v, metric="euclidean"), axis=1)
    parts_sub = parts[vidx]

    vsk = object_model_p.split("/")[-1]

    with open(json_p, "r") as f:
        params = json.load(f)
        rest = EasyDict()
        rest.top = np.array(params["mocap_top"])
        rest.bottom = np.array(params["mocap_bottom"])
        bbox_top = np.array(params["bbox_top"])
        bbox_bottom = np.array(params["bbox_bottom"])
        kp_top = np.array(params["keypoints_top"])
        kp_bottom = np.array(params["keypoints_bottom"])

    np.random.seed(1)

    obj = EasyDict()
    obj.name = vsk
    obj.obj_name = "".join([i for i in vsk if not i.isdigit()])
    obj.v = torch.FloatTensor(mesh_v)
    obj.v_sub = torch.FloatTensor(sub_v)
    obj.f = torch.LongTensor(mesh_f)
    obj.parts = torch.LongTensor(parts)
    obj.parts_sub = torch.LongTensor(parts_sub)

    with open("./data/arctic_data/data/meta/object_meta.json", "r") as f:
        object_meta = json.load(f)
    obj.diameter = torch.FloatTensor(np.array(object_meta[obj.obj_name]["diameter"]))
    obj.bbox_top = torch.FloatTensor(bbox_top)
    obj.bbox_bottom = torch.FloatTensor(bbox_bottom)
    obj.kp_top = torch.FloatTensor(kp_top)
    obj.kp_bottom = torch.FloatTensor(kp_bottom)
    obj.mocap_top = torch.FloatTensor(np.array(params["mocap_top"]))
    obj.mocap_bottom = torch.FloatTensor(np.array(params["mocap_bottom"]))
    return obj


def construct_obj_tensors(object_names):
    obj_list = []
    for k in object_names:
        object_model_p = f"./data/arctic_data/data/meta/object_vtemplates/%s" % (k)
        obj = construct_obj(object_model_p)
        obj_list.append(obj)

    bbox_top_list = []
    bbox_bottom_list = []
    mocap_top_list = []
    mocap_bottom_list = []
    kp_top_list = []
    kp_bottom_list = []
    v_list = []
    v_sub_list = []
    f_list = []
    parts_list = []
    parts_sub_list = []
    diameter_list = []
    for obj in obj_list:
        v_list.append(obj.v)
        v_sub_list.append(obj.v_sub)
        f_list.append(obj.f)

        # root_list.append(obj.root)
        bbox_top_list.append(obj.bbox_top)
        bbox_bottom_list.append(obj.bbox_bottom)
        kp_top_list.append(obj.kp_top)
        kp_bottom_list.append(obj.kp_bottom)
        mocap_top_list.append(obj.mocap_top / 1000)
        mocap_bottom_list.append(obj.mocap_bottom / 1000)
        parts_list.append(obj.parts + 1)
        parts_sub_list.append(obj.parts_sub + 1)
        diameter_list.append(obj.diameter)

    v_list, v_len_list = pad_tensor_list(v_list)
    p_list, p_len_list = pad_tensor_list(parts_list)
    ps_list = torch.stack(parts_sub_list, dim=0)
    assert (p_len_list - v_len_list).sum() == 0

    max_len = v_len_list.max()
    mask = torch.zeros(len(obj_list), max_len)
    for idx, vlen in enumerate(v_len_list):
        mask[idx, :vlen] = 1.0

    v_sub_list = torch.stack(v_sub_list, dim=0)
    diameter_list = torch.stack(diameter_list, dim=0)

    f_list, f_len_list = pad_tensor_list(f_list)

    bbox_top_list = torch.stack(bbox_top_list, dim=0)
    bbox_bottom_list = torch.stack(bbox_bottom_list, dim=0)
    kp_top_list = torch.stack(kp_top_list, dim=0)
    kp_bottom_list = torch.stack(kp_bottom_list, dim=0)

    obj_tensors = {}
    obj_tensors["names"] = object_names
    obj_tensors["parts_ids"] = p_list
    obj_tensors["parts_sub_ids"] = ps_list

    obj_tensors["v"] = v_list.float() / 1000
    obj_tensors["v_sub"] = v_sub_list.float() / 1000
    obj_tensors["v_len"] = v_len_list
    obj_tensors["f"] = f_list
    obj_tensors["f_len"] = f_len_list
    obj_tensors["diameter"] = diameter_list.float()

    obj_tensors["mask"] = mask
    obj_tensors["bbox_top"] = bbox_top_list.float() / 1000
    obj_tensors["bbox_bottom"] = bbox_bottom_list.float() / 1000
    obj_tensors["kp_top"] = kp_top_list.float() / 1000
    obj_tensors["kp_bottom"] = kp_bottom_list.float() / 1000
    obj_tensors["mocap_top"] = mocap_top_list
    obj_tensors["mocap_bottom"] = mocap_bottom_list
    obj_tensors["z_axis"] = torch.FloatTensor(np.array([0, 0, -1])).view(1, 3)
    return obj_tensors
