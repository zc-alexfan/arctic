import torch
import torch.nn as nn

import common.ld_utils as ld_utils
import src.callbacks.process.process_generic as generic
from common.xdict import xdict
from src.nets.hand_heads.hand_hmr import HandHMR
from src.nets.hand_heads.mano_head import MANOHead
from src.nets.obj_heads.obj_head import ArtiHead
from src.nets.obj_heads.obj_hmr import ObjectHMR


class ArcticLSTM(nn.Module):
    def __init__(self, focal_length, img_res, args):
        super().__init__()
        self.args = args
        feat_dim = 2048
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
        self.feat_dim = feat_dim
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=1024,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

    def _fetch_img_feat(self, inputs):
        feat_vec = inputs["img_feat"]
        return feat_vec

    def forward(self, inputs, meta_info):
        window_size = self.args.window_size
        query_names = meta_info["query_names"]
        K = meta_info["intrinsics"]
        device = K.device
        feat_vec = self._fetch_img_feat(inputs)
        feat_vec = feat_vec.view(-1, window_size, self.feat_dim)
        batch_size = feat_vec.shape[0]

        # bidirectional
        h0 = torch.randn(2 * 2, batch_size, self.feat_dim // 2, device=device)
        c0 = torch.randn(2 * 2, batch_size, self.feat_dim // 2, device=device)
        feat_vec, (hn, cn) = self.lstm(feat_vec, (h0, c0))  # batch, seq, 2*dim
        feat_vec = feat_vec.reshape(batch_size * window_size, self.feat_dim)

        hmr_output_r = self.head_r(feat_vec, use_pool=False)
        hmr_output_l = self.head_l(feat_vec, use_pool=False)
        hmr_output_o = self.head_o(feat_vec, use_pool=False)

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
        output = generic.prepare_interfield(output, self.args.max_dist)
        return output
