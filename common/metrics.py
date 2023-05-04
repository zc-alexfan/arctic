import math

import numpy as np
import torch


def compute_v2v_dist_no_reduce(v3d_cam_gt, v3d_cam_pred, is_valid):
    assert isinstance(v3d_cam_gt, list)
    assert isinstance(v3d_cam_pred, list)
    assert len(v3d_cam_gt) == len(v3d_cam_pred)
    assert len(v3d_cam_gt) == len(is_valid)
    v2v = []
    for v_gt, v_pred, valid in zip(v3d_cam_gt, v3d_cam_pred, is_valid):
        if valid:
            dist = ((v_gt - v_pred) ** 2).sum(dim=1).sqrt().cpu().numpy()  # meter
        else:
            dist = None
        v2v.append(dist)
    return v2v


def compute_joint3d_error(joints3d_cam_gt, joints3d_cam_pred, valid_jts):
    valid_jts = valid_jts.view(-1)
    assert joints3d_cam_gt.shape == joints3d_cam_pred.shape
    assert joints3d_cam_gt.shape[0] == valid_jts.shape[0]
    dist = ((joints3d_cam_gt - joints3d_cam_pred) ** 2).sum(dim=2).sqrt()
    invalid_idx = torch.nonzero((1 - valid_jts).long()).view(-1)
    dist[invalid_idx, :] = float("nan")
    dist = dist.cpu().numpy()
    return dist


def compute_mrrpe(root_r_gt, root_l_gt, root_r_pred, root_l_pred, is_valid):
    rel_vec_gt = root_l_gt - root_r_gt
    rel_vec_pred = root_l_pred - root_r_pred

    invalid_idx = torch.nonzero((1 - is_valid).long()).view(-1)
    mrrpe = ((rel_vec_pred - rel_vec_gt) ** 2).sum(dim=1).sqrt()
    mrrpe[invalid_idx] = float("nan")
    mrrpe = mrrpe.cpu().numpy()
    return mrrpe


def compute_arti_deg_error(pred_radian, gt_radian):
    assert pred_radian.shape == gt_radian.shape

    # articulation error in degree
    pred_degree = pred_radian / math.pi * 180  # degree
    gt_degree = gt_radian / math.pi * 180  # degree
    err_deg = torch.abs(pred_degree - gt_degree).tolist()
    return np.array(err_deg, dtype=np.float32)
