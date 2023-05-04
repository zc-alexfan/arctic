import torch.nn as nn

from common.xdict import xdict

l1_loss = nn.L1Loss(reduction="none")
mse_loss = nn.MSELoss(reduction="none")
ce_loss = nn.CrossEntropyLoss(reduction="none")


def dist_loss(loss_dict, pred, gt, meta_info):
    is_valid = gt["is_valid"]
    mask_o = meta_info["mask"]

    # interfield
    loss_ro = mse_loss(pred[f"dist.ro"], gt["dist.ro"])
    loss_lo = mse_loss(pred[f"dist.lo"], gt["dist.lo"])

    pad_olen = min(pred[f"dist.or"].shape[1], gt["dist.or"].shape[1])

    loss_or = mse_loss(pred[f"dist.or"][:, :pad_olen], gt["dist.or"][:, :pad_olen])
    loss_ol = mse_loss(pred[f"dist.ol"][:, :pad_olen], gt["dist.ol"][:, :pad_olen])

    # too many 10cm. Skip them in the loss to prevent overfitting
    bnd = 0.1  # 10cm
    bnd_idx_ro = gt["dist.ro"] == bnd
    bnd_idx_lo = gt["dist.lo"] == bnd
    bnd_idx_or = gt["dist.or"][:, :pad_olen] == bnd
    bnd_idx_ol = gt["dist.ol"][:, :pad_olen] == bnd

    loss_or = loss_or * mask_o * is_valid[:, None]
    loss_ol = loss_ol * mask_o * is_valid[:, None]

    loss_ro = loss_ro * is_valid[:, None]
    loss_lo = loss_lo * is_valid[:, None]

    loss_or[bnd_idx_or] *= 0.1
    loss_ol[bnd_idx_ol] *= 0.1
    loss_ro[bnd_idx_ro] *= 0.1
    loss_lo[bnd_idx_lo] *= 0.1

    weight = 100.0
    loss_dict[f"loss/dist/ro"] = (loss_ro.mean(), weight)
    loss_dict[f"loss/dist/lo"] = (loss_lo.mean(), weight)
    loss_dict[f"loss/dist/or"] = (loss_or.mean(), weight)
    loss_dict[f"loss/dist/ol"] = (loss_ol.mean(), weight)
    return loss_dict


def compute_loss(pred, gt, meta_info, args):
    loss_dict = xdict()
    loss_dict = dist_loss(loss_dict, pred, gt, meta_info)
    return loss_dict
