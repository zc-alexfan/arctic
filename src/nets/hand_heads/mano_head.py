import torch.nn as nn

import common.camera as camera
import common.data_utils as data_utils
import common.rot as rot
import common.transforms as tf
from common.body_models import build_mano_aa
from common.xdict import xdict


class MANOHead(nn.Module):
    def __init__(self, is_rhand, focal_length, img_res):
        super(MANOHead, self).__init__()
        self.mano = build_mano_aa(is_rhand)
        self.add_module("mano", self.mano)
        self.focal_length = focal_length
        self.img_res = img_res
        self.is_rhand = is_rhand

    def forward(self, rotmat, shape, cam, K):
        """
        :param rotmat: rotation in euler angles format (N,J,3,3)
        :param shape: smpl betas
        :param cam: weak perspective camera
        :param normalize_joints2d: bool, normalize joints between -1, 1 if true
        :return: dict with keys 'vertices', 'joints3d', 'joints2d' if cam is True
        """

        rotmat_original = rotmat.clone()
        rotmat = rot.matrix_to_axis_angle(rotmat.reshape(-1, 3, 3)).reshape(-1, 48)

        mano_output = self.mano(
            betas=shape,
            hand_pose=rotmat[:, 3:],
            global_orient=rotmat[:, :3],
        )
        output = xdict()

        avg_focal_length = (K[:, 0, 0] + K[:, 1, 1]) / 2.0
        cam_t = camera.weak_perspective_to_perspective_torch(
            cam, focal_length=avg_focal_length, img_res=self.img_res, min_s=0.1
        )

        joints3d_cam = mano_output.joints + cam_t[:, None, :]
        v3d_cam = mano_output.vertices + cam_t[:, None, :]

        joints2d = tf.project2d_batch(K, joints3d_cam)
        joints2d = data_utils.normalize_kp2d(joints2d, self.img_res)

        output["cam_t.wp"] = cam
        output["cam_t"] = cam_t
        output["joints3d"] = mano_output.joints
        output["vertices"] = mano_output.vertices
        output["j3d.cam"] = joints3d_cam
        output["v3d.cam"] = v3d_cam
        output["j2d.norm"] = joints2d
        output["beta"] = shape
        output["pose"] = rotmat_original

        postfix = ".r" if self.is_rhand else ".l"
        output_pad = output.postfix(postfix)
        return output_pad
