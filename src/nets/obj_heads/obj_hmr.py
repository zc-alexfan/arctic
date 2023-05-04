import torch
import torch.nn as nn

from common.xdict import xdict
from src.nets.hmr_layer import HMRLayer


class ObjectHMR(nn.Module):
    def __init__(self, feat_dim, n_iter):
        super().__init__()

        obj_specs = {"rot": 3, "cam_t/wp": 3, "radian": 1}
        self.hmr_layer = HMRLayer(feat_dim, 1024, obj_specs)

        self.cam_init = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

        self.obj_specs = obj_specs
        self.n_iter = n_iter
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def init_vector_dict(self, features):
        batch_size = features.shape[0]
        dev = features.device
        init_rot = torch.zeros(batch_size, 3)
        init_angle = torch.zeros(batch_size, 1)
        init_transl = self.cam_init(features)

        out = {}
        out["rot"] = init_rot
        out["radian"] = init_angle
        out["cam_t/wp"] = init_transl
        out = xdict(out).to(dev)
        return out

    def forward(self, features, use_pool=True):
        if use_pool:
            feat = self.avgpool(features)
            feat = feat.view(feat.size(0), -1)
        else:
            feat = features

        init_vdict = self.init_vector_dict(feat)
        init_cam_t = init_vdict["cam_t/wp"].clone()
        pred_vdict = self.hmr_layer(feat, init_vdict, self.n_iter)
        pred_vdict["cam_t.wp.init"] = init_cam_t
        pred_vdict = pred_vdict.replace_keys("/", ".")
        return pred_vdict
