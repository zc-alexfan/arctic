import torch
import torch.nn as nn
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix

from src.utils.loss_modules import (
    compute_contact_devi_loss,
    hand_kp3d_loss,
    joints_loss,
    mano_loss,
    object_kp3d_loss,
    vector_loss,
)

l1_loss = nn.L1Loss(reduction="none")
mse_loss = nn.MSELoss(reduction="none")


def compute_loss(pred, gt, meta_info, args):
    # unpacking pred and gt
    pred_betas_r = pred["mano.beta.r"]
    pred_rotmat_r = pred["mano.pose.r"]
    pred_joints_r = pred["mano.j3d.cam.r"]
    pred_projected_keypoints_2d_r = pred["mano.j2d.norm.r"]
    pred_betas_l = pred["mano.beta.l"]
    pred_rotmat_l = pred["mano.pose.l"]
    pred_joints_l = pred["mano.j3d.cam.l"]
    pred_projected_keypoints_2d_l = pred["mano.j2d.norm.l"]
    pred_kp2d_o = pred["object.kp2d.norm"]
    pred_kp3d_o = pred["object.kp3d.cam"]
    pred_rot = pred["object.rot"].view(-1, 3).float()
    pred_radian = pred["object.radian"].view(-1).float()

    gt_pose_r = gt["mano.pose.r"]
    gt_betas_r = gt["mano.beta.r"]
    gt_joints_r = gt["mano.j3d.cam.r"]
    gt_keypoints_2d_r = gt["mano.j2d.norm.r"]
    gt_pose_l = gt["mano.pose.l"]
    gt_betas_l = gt["mano.beta.l"]
    gt_joints_l = gt["mano.j3d.cam.l"]
    gt_keypoints_2d_l = gt["mano.j2d.norm.l"]
    gt_kp2d_o = torch.cat((gt["object.kp2d.norm.t"], gt["object.kp2d.norm.b"]), dim=1)
    gt_kp3d_o = gt["object.kp3d.cam"]
    gt_rot = gt["object.rot"].view(-1, 3).float()
    gt_radian = gt["object.radian"].view(-1).float()

    is_valid = gt["is_valid"]
    right_valid = gt["right_valid"]
    left_valid = gt["left_valid"]
    joints_valid_r = gt["joints_valid_r"]
    joints_valid_l = gt["joints_valid_l"]

    # reshape
    gt_pose_r = axis_angle_to_matrix(gt_pose_r.reshape(-1, 3)).reshape(-1, 16, 3, 3)
    gt_pose_l = axis_angle_to_matrix(gt_pose_l.reshape(-1, 3)).reshape(-1, 16, 3, 3)

    # Compute loss on MANO parameters
    loss_regr_pose_r, loss_regr_betas_r = mano_loss(
        pred_rotmat_r,
        pred_betas_r,
        gt_pose_r,
        gt_betas_r,
        criterion=mse_loss,
        is_valid=right_valid,
    )
    loss_regr_pose_l, loss_regr_betas_l = mano_loss(
        pred_rotmat_l,
        pred_betas_l,
        gt_pose_l,
        gt_betas_l,
        criterion=mse_loss,
        is_valid=left_valid,
    )

    # Compute 2D reprojection loss for the keypoints
    loss_keypoints_r = joints_loss(
        pred_projected_keypoints_2d_r,
        gt_keypoints_2d_r,
        criterion=mse_loss,
        jts_valid=joints_valid_r,
    )
    loss_keypoints_l = joints_loss(
        pred_projected_keypoints_2d_l,
        gt_keypoints_2d_l,
        criterion=mse_loss,
        jts_valid=joints_valid_l,
    )

    loss_keypoints_o = vector_loss(
        pred_kp2d_o, gt_kp2d_o, criterion=mse_loss, is_valid=is_valid
    )

    # Compute 3D keypoint loss
    loss_keypoints_3d_r = hand_kp3d_loss(
        pred_joints_r, gt_joints_r, mse_loss, joints_valid_r
    )
    loss_keypoints_3d_l = hand_kp3d_loss(
        pred_joints_l, gt_joints_l, mse_loss, joints_valid_l
    )
    loss_keypoints_3d_o = object_kp3d_loss(pred_kp3d_o, gt_kp3d_o, mse_loss, is_valid)

    loss_radian = vector_loss(pred_radian, gt_radian, mse_loss, is_valid)
    loss_rot = vector_loss(pred_rot, gt_rot, mse_loss, is_valid)
    loss_transl_l = vector_loss(
        pred["mano.cam_t.wp.l"] - pred["mano.cam_t.wp.r"],
        gt["mano.cam_t.wp.l"] - gt["mano.cam_t.wp.r"],
        mse_loss,
        right_valid * left_valid,
    )
    loss_transl_o = vector_loss(
        pred["object.cam_t.wp"] - pred["mano.cam_t.wp.r"],
        gt["object.cam_t.wp"] - gt["mano.cam_t.wp.r"],
        mse_loss,
        right_valid * is_valid,
    )

    loss_cam_t_r = vector_loss(
        pred["mano.cam_t.wp.r"],
        gt["mano.cam_t.wp.r"],
        mse_loss,
        right_valid,
    )
    loss_cam_t_l = vector_loss(
        pred["mano.cam_t.wp.l"],
        gt["mano.cam_t.wp.l"],
        mse_loss,
        left_valid,
    )
    loss_cam_t_o = vector_loss(
        pred["object.cam_t.wp"], gt["object.cam_t.wp"], mse_loss, is_valid
    )

    loss_cam_t_r += vector_loss(
        pred["mano.cam_t.wp.init.r"],
        gt["mano.cam_t.wp.r"],
        mse_loss,
        right_valid,
    )
    loss_cam_t_l += vector_loss(
        pred["mano.cam_t.wp.init.l"],
        gt["mano.cam_t.wp.l"],
        mse_loss,
        left_valid,
    )
    loss_cam_t_o += vector_loss(
        pred["object.cam_t.wp.init"],
        gt["object.cam_t.wp"],
        mse_loss,
        is_valid,
    )
    cd_ro, cd_lo = compute_contact_devi_loss(pred, gt)

    loss_dict = {
        "loss/mano/cam_t/r": (loss_cam_t_r, 1.0),
        "loss/mano/cam_t/l": (loss_cam_t_l, 1.0),
        "loss/object/cam_t": (loss_cam_t_o, 1.0),
        "loss/mano/kp2d/r": (loss_keypoints_r, 5.0),
        "loss/mano/kp3d/r": (loss_keypoints_3d_r, 5.0),
        "loss/mano/pose/r": (loss_regr_pose_r, 10.0),
        "loss/mano/beta/r": (loss_regr_betas_r, 0.001),
        "loss/mano/kp2d/l": (loss_keypoints_l, 5.0),
        "loss/mano/kp3d/l": (loss_keypoints_3d_l, 5.0),
        "loss/mano/pose/l": (loss_regr_pose_l, 10.0),
        "loss/cd": (cd_ro + cd_lo, 1.0),
        "loss/mano/transl/l": (loss_transl_l, 1.0),
        "loss/mano/beta/l": (loss_regr_betas_l, 0.001),
        "loss/object/kp2d": (loss_keypoints_o, 1.0),
        "loss/object/kp3d": (loss_keypoints_3d_o, 5.0),
        "loss/object/radian": (loss_radian, 1.0),
        "loss/object/rot": (loss_rot, 1.0),
        "loss/object/transl": (loss_transl_o, 1.0),
    }
    return loss_dict
