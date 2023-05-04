import os
import os.path as op
from glob import glob

import numpy as np
from loguru import logger

import common.data_utils as data_utils
from common.sys_utils import copy_repo


def transform_bbox_for_speedup(
    speedup,
    is_egocam,
    _bbox_crop,
    ego_image_scale,
):
    bbox_crop = np.array(_bbox_crop)
    # bbox is normalized in scale

    if speedup:
        if is_egocam:
            bbox_crop = [num * ego_image_scale for num in bbox_crop]
        else:
            # change to new coord system
            bbox_crop[0] = 500
            bbox_crop[1] = 500
            bbox_crop[2] = 1000 / (1.5 * 200)

    # bbox is normalized in scale
    return bbox_crop


def transform_2d_for_speedup(
    speedup,
    is_egocam,
    _joints2d_r,
    _joints2d_l,
    _kp2d_b,
    _kp2d_t,
    _bbox2d_b,
    _bbox2d_t,
    _bbox_crop,
    ego_image_scale,
):
    joints2d_r = np.copy(_joints2d_r)
    joints2d_l = np.copy(_joints2d_l)
    kp2d_b = np.copy(_kp2d_b)
    kp2d_t = np.copy(_kp2d_t)
    bbox2d_b = np.copy(_bbox2d_b)
    bbox2d_t = np.copy(_bbox2d_t)
    bbox_crop = np.array(_bbox_crop)
    # bbox is normalized in scale

    if speedup:
        if is_egocam:
            joints2d_r[:, :2] *= ego_image_scale
            joints2d_l[:, :2] *= ego_image_scale
            kp2d_b[:, :2] *= ego_image_scale
            kp2d_t[:, :2] *= ego_image_scale
            bbox2d_b[:, :2] *= ego_image_scale
            bbox2d_t[:, :2] *= ego_image_scale

            bbox_crop = [num * ego_image_scale for num in bbox_crop]
        else:
            # change to new coord system
            joints2d_r = data_utils.transform_kp2d(joints2d_r, bbox_crop)
            joints2d_l = data_utils.transform_kp2d(joints2d_l, bbox_crop)
            kp2d_b = data_utils.transform_kp2d(kp2d_b, bbox_crop)
            kp2d_t = data_utils.transform_kp2d(kp2d_t, bbox_crop)
            bbox2d_b = data_utils.transform_kp2d(bbox2d_b, bbox_crop)
            bbox2d_t = data_utils.transform_kp2d(bbox2d_t, bbox_crop)

            bbox_crop[0] = 500
            bbox_crop[1] = 500
            bbox_crop[2] = 1000 / (1.5 * 200)

    # bbox is normalized in scale
    return (
        joints2d_r,
        joints2d_l,
        kp2d_b,
        kp2d_t,
        bbox2d_b,
        bbox2d_t,
        bbox_crop,
    )


def copy_repo_arctic(exp_key):
    dst_folder = f"/is/cluster/work/fzicong/chiral_data/cache/logs/{exp_key}/repo"

    if not op.exists(dst_folder):
        logger.info("Copying repo")
        src_files = glob("./*")
        os.makedirs(dst_folder)
        filter_keywords = [".ipynb", ".obj", ".pt", "run_scripts", ".sub", ".txt"]
        copy_repo(src_files, dst_folder, filter_keywords)
        logger.info("Done")


def get_num_images(split, num_images):
    if split in ["train", "val", "test"]:
        return num_images

    if split == "smalltrain":
        return 100000

    if split == "tinytrain":
        return 12000

    if split == "minitrain":
        return 300

    if split == "smallval":
        return 12000

    if split == "tinyval":
        return 500

    if split == "minival":
        return 80

    if split == "smalltest":
        return 12000

    if split == "tinytest":
        return 6000

    if split == "minitest":
        return 200

    assert False, f"Invalid split {split}"


def pad_jts2d(jts):
    num_jts = jts.shape[0]
    jts_pad = np.ones((num_jts, 3))
    jts_pad[:, :2] = jts
    return jts_pad


def get_valid(data_2d, data_cam, vidx, view_idx, imgname):
    assert (
        vidx < data_2d["joints.right"].shape[0]
    ), "The requested vidx does not exist in annotation"
    is_valid = data_cam["is_valid"][vidx, view_idx]
    right_valid = data_cam["right_valid"][vidx, view_idx]
    left_valid = data_cam["left_valid"][vidx, view_idx]
    return vidx, is_valid, right_valid, left_valid


def downsample(fnames, split):
    if "small" not in split and "mini" not in split and "tiny" not in split:
        return fnames
    import random

    random.seed(1)
    assert (
        random.randint(0, 100) == 17
    ), "Same seed but different results; Subsampling might be different."

    num_samples = get_num_images(split, len(fnames))
    curr_keys = random.sample(fnames, num_samples)
    return curr_keys
