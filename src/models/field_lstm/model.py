import torch
import torch.nn as nn

from common.xdict import xdict
from src.models.field_sf.model import RegressHead, Upsampler
from src.nets.backbone.utils import get_backbone_info
from src.nets.obj_heads.obj_head import ArtiHead
from src.nets.pointnet import PointNetfeat


class FieldLSTM(nn.Module):
    def __init__(self, backbone, focal_length, img_res, window_size):
        super().__init__()
        assert backbone in ["resnet18", "resnet50"]
        feat_dim = get_backbone_info(backbone)["n_output_channels"]
        self.arti_head = ArtiHead(focal_length=focal_length, img_res=img_res)

        img_down_dim = 512
        img_mid_dim = 512
        pt_out_dim = 512
        self.down = nn.Sequential(
            nn.Linear(feat_dim, img_mid_dim),
            nn.ReLU(),
            nn.Linear(img_mid_dim, img_down_dim),
            nn.ReLU(),
        )  # downsize image features

        pt_shallow_dim = 512
        pt_mid_dim = 512
        self.point_backbone = PointNetfeat(
            input_dim=3 + img_down_dim,
            shallow_dim=pt_shallow_dim,
            mid_dim=pt_mid_dim,
            out_dim=pt_out_dim,
        )
        pts_dim = pt_shallow_dim + pt_out_dim
        self.dist_head_or = RegressHead(pts_dim)
        self.dist_head_ol = RegressHead(pts_dim)
        self.dist_head_ro = RegressHead(pts_dim)
        self.dist_head_lo = RegressHead(pts_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.num_v_sub = 195  # mano subsampled
        self.num_v_o_sub = 300 * 2  # object subsampled
        self.num_v_o = 4000  # object
        self.upsampling_r = Upsampler(self.num_v_sub, 778)
        self.upsampling_l = Upsampler(self.num_v_sub, 778)
        self.upsampling_o = Upsampler(self.num_v_o_sub, self.num_v_o)
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=1024,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        self.feat_dim = feat_dim
        self.window_size = window_size

    def forward(self, inputs, meta_info):
        window_size = self.window_size
        device = meta_info["v0.r"].device

        feat_vec = inputs["img_feat"].view(-1, window_size, self.feat_dim)
        batch_size = feat_vec.shape[0]

        points_r = meta_info["v0.r"].permute(0, 2, 1)[:, :, 21:]
        points_l = meta_info["v0.l"].permute(0, 2, 1)[:, :, 21:]
        points_o = meta_info["v0.o"].permute(0, 2, 1)
        points_all = torch.cat((points_r, points_l, points_o), dim=2)

        # bidirectional
        h0 = torch.randn(2 * 2, batch_size, self.feat_dim // 2, device=device)
        c0 = torch.randn(2 * 2, batch_size, self.feat_dim // 2, device=device)
        feat_vec, (hn, cn) = self.lstm(feat_vec, (h0, c0))  # batch, seq, 2*dim
        feat_vec = feat_vec.reshape(batch_size * window_size, self.feat_dim)

        img_feat = self.down(feat_vec)
        num_mano_pts = points_r.shape[2]
        num_object_pts = points_o.shape[2]

        img_feat_all = img_feat[:, :, None].repeat(
            1, 1, num_mano_pts * 2 + num_object_pts
        )
        pts_all_feat = self.point_backbone(
            torch.cat((points_all, img_feat_all), dim=1)
        )[0]
        pts_r_feat, pts_l_feat, pts_o_feat = torch.split(
            pts_all_feat, [num_mano_pts, num_mano_pts, num_object_pts], dim=2
        )

        dist_ro = self.dist_head_ro(pts_r_feat)
        dist_lo = self.dist_head_lo(pts_l_feat)
        dist_or = self.dist_head_or(pts_o_feat)
        dist_ol = self.dist_head_ol(pts_o_feat)

        dist_ro = self.upsampling_r(dist_ro[:, :, None])[:, :, 0]
        dist_lo = self.upsampling_l(dist_lo[:, :, None])[:, :, 0]
        dist_or = self.upsampling_o(dist_or[:, :, None])[:, :, 0]
        dist_ol = self.upsampling_o(dist_ol[:, :, None])[:, :, 0]

        out = xdict()
        out["dist.ro"] = dist_ro
        out["dist.lo"] = dist_lo
        out["dist.or"] = dist_or
        out["dist.ol"] = dist_ol
        return out
