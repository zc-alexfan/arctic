import torch.nn as nn

import common.ld_utils as ld_utils
from common.xdict import xdict
from src.nets.backbone.utils import get_backbone_info
from src.nets.hand_heads.hand_hmr import HandHMR
from src.nets.hand_heads.mano_head import MANOHead
from src.nets.obj_heads.obj_head import ArtiHead
from src.nets.obj_heads.obj_hmr import ObjectHMR


class ArcticSF(nn.Module):
    def __init__(self, backbone, focal_length, img_res, args):
        super(ArcticSF, self).__init__()
        self.args = args
        if backbone == "resnet50":
            from src.nets.backbone.resnet import resnet50 as resnet
        elif backbone == "resnet18":
            from src.nets.backbone.resnet import resnet18 as resnet
        else:
            assert False
        self.backbone = resnet(pretrained=True)
        feat_dim = get_backbone_info(backbone)["n_output_channels"]
        self.head_r = HandHMR(feat_dim, is_rhand=True, n_iter=3)
        self.head_l = HandHMR(feat_dim, is_rhand=False, n_iter=3)

        self.head_o = ObjectHMR(feat_dim, n_iter=3)

        self.mano_r = MANOHead(
            is_rhand=True, focal_length=focal_length, img_res=img_res
        )

        self.mano_l = MANOHead(
            is_rhand=False, focal_length=focal_length, img_res=img_res
        )

        self.arti_head = ArtiHead(focal_length=focal_length, img_res=img_res)
        self.mode = "train"
        self.img_res = img_res
        self.focal_length = focal_length

    def forward(self, inputs, meta_info):
        images = inputs["img"]
        query_names = meta_info["query_names"]
        K = meta_info["intrinsics"]
        features = self.backbone(images)
        feat_vec = features.view(features.shape[0], features.shape[1], -1).sum(dim=2)

        hmr_output_r = self.head_r(features)
        hmr_output_l = self.head_l(features)
        hmr_output_o = self.head_o(features)

        # weak perspective
        root_r = hmr_output_r["cam_t.wp"]
        root_l = hmr_output_l["cam_t.wp"]
        root_o = hmr_output_o["cam_t.wp"]

        mano_output_r = self.mano_r(
            rotmat=hmr_output_r["pose"],
            shape=hmr_output_r["shape"],
            K=K,
            cam=root_r,
        )

        mano_output_l = self.mano_l(
            rotmat=hmr_output_l["pose"],
            shape=hmr_output_l["shape"],
            K=K,
            cam=root_l,
        )

        # fwd mesh when in val or vis
        arti_output = self.arti_head(
            rot=hmr_output_o["rot"],
            angle=hmr_output_o["radian"],
            query_names=query_names,
            cam=root_o,
            K=K,
        )

        root_r_init = hmr_output_r["cam_t.wp.init"]
        root_l_init = hmr_output_l["cam_t.wp.init"]
        root_o_init = hmr_output_o["cam_t.wp.init"]
        mano_output_r["cam_t.wp.init.r"] = root_r_init
        mano_output_l["cam_t.wp.init.l"] = root_l_init
        arti_output["cam_t.wp.init"] = root_o_init

        mano_output_r = ld_utils.prefix_dict(mano_output_r, "mano.")
        mano_output_l = ld_utils.prefix_dict(mano_output_l, "mano.")
        arti_output = ld_utils.prefix_dict(arti_output, "object.")
        output = xdict()
        output.merge(mano_output_r)
        output.merge(mano_output_l)
        output.merge(arti_output)
        output["feat_vec"] = feat_vec.cpu().detach()
        return output
