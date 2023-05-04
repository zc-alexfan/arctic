"""
This file contains functions that are used to perform data augmentation.
"""
import cv2
import numpy as np
import torch
from loguru import logger


def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + 0.5)
    t[1, 2] = res[0] * (-float(center[1]) / h + 0.5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.0]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(
    c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False
):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans


def generate_patch_image(
    cvimg,
    bbox,
    scale,
    rot,
    out_shape,
    interpl_strategy,
    gauss_kernel=5,
    gauss_sigma=8.0,
):
    img = cvimg.copy()

    bb_c_x = float(bbox[0])
    bb_c_y = float(bbox[1])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    trans = gen_trans_from_patch_cv(
        bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot
    )

    # anti-aliasing
    blur = cv2.GaussianBlur(img, (gauss_kernel, gauss_kernel), gauss_sigma)
    img_patch = cv2.warpAffine(
        blur, trans, (int(out_shape[1]), int(out_shape[0])), flags=interpl_strategy
    )
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(
        bb_c_x,
        bb_c_y,
        bb_width,
        bb_height,
        out_shape[1],
        out_shape[0],
        scale,
        rot,
        inv=True,
    )

    return img_patch, trans, inv_trans


def augm_params(is_train, flip_prob, noise_factor, rot_factor, scale_factor):
    """Get augmentation parameters."""
    flip = 0  # flipping
    pn = np.ones(3)  # per channel pixel-noise
    rot = 0  # rotation
    sc = 1  # scaling
    if is_train:
        # We flip with probability 1/2
        if np.random.uniform() <= flip_prob:
            flip = 1
            assert False, "Flipping not supported"

        # Each channel is multiplied with a number
        # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
        pn = np.random.uniform(1 - noise_factor, 1 + noise_factor, 3)

        # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
        rot = min(
            2 * rot_factor,
            max(
                -2 * rot_factor,
                np.random.randn() * rot_factor,
            ),
        )

        # The scale is multiplied with a number
        # in the area [1-scaleFactor,1+scaleFactor]
        sc = min(
            1 + scale_factor,
            max(
                1 - scale_factor,
                np.random.randn() * scale_factor + 1,
            ),
        )
        # but it is zero with probability 3/5
        if np.random.uniform() <= 0.6:
            rot = 0

    augm_dict = {}
    augm_dict["flip"] = flip
    augm_dict["pn"] = pn
    augm_dict["rot"] = rot
    augm_dict["sc"] = sc
    return augm_dict


def rgb_processing(is_train, rgb_img, center, bbox_dim, augm_dict, img_res):
    rot = augm_dict["rot"]
    sc = augm_dict["sc"]
    pn = augm_dict["pn"]
    scale = sc * bbox_dim

    crop_dim = int(scale * 200)
    # faster cropping!!
    rgb_img = generate_patch_image(
        rgb_img,
        [center[0], center[1], crop_dim, crop_dim],
        1.0,
        rot,
        [img_res, img_res],
        cv2.INTER_CUBIC,
    )[0]

    # in the rgb image we add pixel noise in a channel-wise manner
    rgb_img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 0] * pn[0]))
    rgb_img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 1] * pn[1]))
    rgb_img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 2] * pn[2]))
    rgb_img = np.transpose(rgb_img.astype("float32"), (2, 0, 1)) / 255.0
    return rgb_img


def transform_kp2d(kp2d, bbox):
    # bbox: (cx, cy, scale) in the original image space
    # scale is normalized
    assert isinstance(kp2d, np.ndarray)
    assert len(kp2d.shape) == 2
    cx, cy, scale = bbox
    s = 200 * scale  # to px
    cap_dim = 1000  # px
    factor = cap_dim / (1.5 * s)
    kp2d_cropped = np.copy(kp2d)
    kp2d_cropped[:, 0] -= cx - 1.5 / 2 * s
    kp2d_cropped[:, 1] -= cy - 1.5 / 2 * s
    kp2d_cropped[:, 0] *= factor
    kp2d_cropped[:, 1] *= factor
    return kp2d_cropped


def j2d_processing(kp, center, bbox_dim, augm_dict, img_res):
    """Process gt 2D keypoints and apply all augmentation transforms."""
    scale = augm_dict["sc"] * bbox_dim
    rot = augm_dict["rot"]

    nparts = kp.shape[0]
    for i in range(nparts):
        kp[i, 0:2] = transform(
            kp[i, 0:2] + 1,
            center,
            scale,
            [img_res, img_res],
            rot=rot,
        )
    # convert to normalized coordinates
    kp = normalize_kp2d_np(kp, img_res)
    kp = kp.astype("float32")
    return kp


def pose_processing(pose, augm_dict):
    """Process SMPL theta parameters  and apply all augmentation transforms."""
    rot = augm_dict["rot"]
    # rotation or the pose parameters
    pose[:3] = rot_aa(pose[:3], rot)
    # flip the pose parameters
    # (72),float
    pose = pose.astype("float32")
    return pose


def rot_aa(aa, rot):
    """Rotate axis angle parameters."""
    # pose parameters
    R = np.array(
        [
            [np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
            [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
            [0, 0, 1],
        ]
    )
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R, per_rdg))
    aa = (resrot.T)[0]
    return aa


def denormalize_images(images):
    images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(
        1, 3, 1, 1
    )
    images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(
        1, 3, 1, 1
    )
    return images


def read_img(img_fn, dummy_shape):
    try:
        cv_img = _read_img(img_fn)
    except:
        logger.warning(f"Unable to load {img_fn}")
        cv_img = np.zeros(dummy_shape, dtype=np.float32)
        return cv_img, False
    return cv_img, True


def _read_img(img_fn):
    img = cv2.cvtColor(cv2.imread(img_fn), cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)


def normalize_kp2d_np(kp2d: np.ndarray, img_res):
    assert kp2d.shape[1] == 3
    kp2d_normalized = kp2d.copy()
    kp2d_normalized[:, :2] = 2.0 * kp2d[:, :2] / img_res - 1.0
    return kp2d_normalized


def unnormalize_2d_kp(kp_2d_np: np.ndarray, res):
    assert kp_2d_np.shape[1] == 3
    kp_2d = np.copy(kp_2d_np)
    kp_2d[:, :2] = 0.5 * res * (kp_2d[:, :2] + 1)
    return kp_2d


def normalize_kp2d(kp2d: torch.Tensor, img_res):
    assert len(kp2d.shape) == 3
    kp2d_normalized = kp2d.clone()
    kp2d_normalized[:, :, :2] = 2.0 * kp2d[:, :, :2] / img_res - 1.0
    return kp2d_normalized


def unormalize_kp2d(kp2d_normalized: torch.Tensor, img_res):
    assert len(kp2d_normalized.shape) == 3
    assert kp2d_normalized.shape[2] == 2
    kp2d = kp2d_normalized.clone()
    kp2d = 0.5 * img_res * (kp2d + 1)
    return kp2d


def get_wp_intrix(fixed_focal: float, img_res):
    # consruct weak perspective on patch
    camera_center = np.array([img_res // 2, img_res // 2])
    intrx = torch.zeros([3, 3])
    intrx[0, 0] = fixed_focal
    intrx[1, 1] = fixed_focal
    intrx[2, 2] = 1.0
    intrx[0, -1] = camera_center[0]
    intrx[1, -1] = camera_center[1]
    return intrx


def get_aug_intrix(
    intrx, fixed_focal: float, img_res, use_gt_k, bbox_cx, bbox_cy, scale
):
    """
    This function returns camera intrinsics under scaling.
    If use_gt_k, the GT K is used, but scaled based on the amount of scaling in the patch.
    Else, we construct an intrinsic camera with a fixed focal length and fixed camera center.
    """

    if not use_gt_k:
        # consruct weak perspective on patch
        intrx = get_wp_intrix(fixed_focal, img_res)
    else:
        # update the GT intrinsics (full image space)
        # such that it matches the scale of the patch

        dim = scale * 200.0  # bbox size
        k_scale = float(img_res) / dim  # resized_dim / bbox_size in full image space
        """
        # x1 and y1: top-left corner of bbox
        intrinsics after data augmentation
        fx' = k*fx
        fy' = k*fy
        cx' = k*(cx - x1)
        cy' = k*(cy - y1)
        """
        intrx[0, 0] *= k_scale  # k*fx
        intrx[1, 1] *= k_scale  # k*fy
        intrx[0, 2] -= bbox_cx - dim / 2.0
        intrx[1, 2] -= bbox_cy - dim / 2.0
        intrx[0, 2] *= k_scale
        intrx[1, 2] *= k_scale
    return intrx
