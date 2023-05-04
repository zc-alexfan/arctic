import pytorch3d.transforms.rotation_conversions as rot_conv
import torch
import torch.nn as nn

from common.xdict import xdict
from src.nets.hmr_layer import HMRLayer


class HandHMR(nn.Module):
    def __init__(self, feat_dim, is_rhand, n_iter):
        super().__init__()
        self.is_rhand = is_rhand

        hand_specs = {"pose_6d": 6 * 16, "cam_t/wp": 3, "shape": 10}
        self.hmr_layer = HMRLayer(feat_dim, 1024, hand_specs)

        self.cam_init = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

        self.hand_specs = hand_specs
        self.n_iter = n_iter
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def init_vector_dict(self, features):
        batch_size = features.shape[0]
        dev = features.device
        init_pose = (
            rot_conv.matrix_to_rotation_6d(
                rot_conv.axis_angle_to_matrix(torch.zeros(16, 3))
            )
            .reshape(1, -1)
            .repeat(batch_size, 1)
        )
        init_shape = torch.zeros(1, 10).repeat(batch_size, 1)
        init_transl = self.cam_init(features)

        out = {}
        out["pose_6d"] = init_pose
        out["shape"] = init_shape
        out["cam_t/wp"] = init_transl
        out = xdict(out).to(dev)
        return out

    def forward(self, features, use_pool=True):
        batch_size = features.shape[0]
        if use_pool:
            feat = self.avgpool(features)
            feat = feat.view(feat.size(0), -1)
        else:
            feat = features

        init_vdict = self.init_vector_dict(feat)
        init_cam_t = init_vdict["cam_t/wp"].clone()
        pred_vdict = self.hmr_layer(feat, init_vdict, self.n_iter)

        pred_rotmat = rot_conv.rotation_6d_to_matrix(
            pred_vdict["pose_6d"].reshape(-1, 6)
        ).view(batch_size, 16, 3, 3)

        pred_vdict["pose"] = pred_rotmat
        pred_vdict["cam_t.wp.init"] = init_cam_t
        pred_vdict = pred_vdict.replace_keys("/", ".")
        return pred_vdict
