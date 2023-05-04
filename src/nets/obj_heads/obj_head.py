import torch.nn as nn

import common.camera as camera
import common.data_utils as data_utils
import common.transforms as tf
from common.object_tensors import ObjectTensors
from common.xdict import xdict


class ArtiHead(nn.Module):
    def __init__(self, focal_length, img_res):
        super().__init__()
        self.object_tensors = ObjectTensors()
        self.focal_length = focal_length
        self.img_res = img_res

    def forward(
        self,
        rot,
        angle,
        query_names,
        cam,
        K,
        transl=None,
    ):
        if self.object_tensors.dev != rot.device:
            self.object_tensors.to(rot.device)

        out = self.object_tensors.forward(angle.view(-1, 1), rot, transl, query_names)

        # after adding relative transl
        bbox3d = out["bbox3d"]
        kp3d = out["kp3d"]

        # right hand translation
        avg_focal_length = (K[:, 0, 0] + K[:, 1, 1]) / 2.0
        cam_t = camera.weak_perspective_to_perspective_torch(
            cam, focal_length=avg_focal_length, img_res=self.img_res, min_s=0.1
        )

        # camera coord
        bbox3d_cam = bbox3d + cam_t[:, None, :]
        kp3d_cam = kp3d + cam_t[:, None, :]

        # 2d keypoints
        kp2d = tf.project2d_batch(K, kp3d_cam)
        bbox2d = tf.project2d_batch(K, bbox3d_cam)

        kp2d = data_utils.normalize_kp2d(kp2d, self.img_res)
        bbox2d = data_utils.normalize_kp2d(bbox2d, self.img_res)
        num_kps = kp2d.shape[1] // 2

        output = xdict()
        output["rot"] = rot
        if transl is not None:
            # relative transl
            output["transl"] = transl  # mete

        output["cam_t.wp"] = cam
        output["cam_t"] = cam_t
        output["kp3d"] = kp3d
        output["bbox3d"] = bbox3d
        output["bbox3d.cam"] = bbox3d_cam
        output["kp3d.cam"] = kp3d_cam
        output["kp2d.norm"] = kp2d
        output["kp2d.norm.t"] = kp2d[:, :num_kps]
        output["kp2d.norm.b"] = kp2d[:, num_kps:]
        output["bbox2d.norm.t"] = bbox2d[:, :8]
        output["bbox2d.norm.b"] = bbox2d[:, 8:]
        output["radian"] = angle

        output["v.cam"] = out["v"] + cam_t[:, None, :]
        output["v_len"] = out["v_len"]
        output["f"] = out["f"]
        output["f_len"] = out["f_len"]

        return output
