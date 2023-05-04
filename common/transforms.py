import numpy as np
import torch

import common.data_utils as data_utils
from common.np_utils import permute_np

"""
Useful geometric operations, e.g. Perspective projection and a differentiable Rodrigues formula
Parts of the code are taken from https://github.com/MandyMo/pytorch_HMR
"""


def to_xy_batch(x_homo):
    assert isinstance(x_homo, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert x_homo.shape[2] == 3
    assert len(x_homo.shape) == 3
    batch_size = x_homo.shape[0]
    num_pts = x_homo.shape[1]
    x = torch.ones(batch_size, num_pts, 2, device=x_homo.device)
    x = x_homo[:, :, :2] / x_homo[:, :, 2:3]
    return x


# VR Distortion Correction Using Vertex Displacement
# https://stackoverflow.com/questions/44489686/camera-lens-distortion-in-opengl
def distort_pts3d_all(_pts_cam, dist_coeffs):
    # egocentric cameras commonly has heavy distortion
    # this function transform points in the undistorted camera coord
    # to distorted camera coord such that the 2d projection can match the pixels.
    pts_cam = _pts_cam.clone().double()
    z = pts_cam[:, :, 2]

    z_inv = 1 / z

    x1 = pts_cam[:, :, 0] * z_inv
    y1 = pts_cam[:, :, 1] * z_inv

    # precalculations
    x1_2 = x1 * x1
    y1_2 = y1 * y1
    x1_y1 = x1 * y1
    r2 = x1_2 + y1_2
    r4 = r2 * r2
    r6 = r4 * r2

    r_dist = (1 + dist_coeffs[0] * r2 + dist_coeffs[1] * r4 + dist_coeffs[4] * r6) / (
        1 + dist_coeffs[5] * r2 + dist_coeffs[6] * r4 + dist_coeffs[7] * r6
    )

    # full (rational + tangential) distortion
    x2 = x1 * r_dist + 2 * dist_coeffs[2] * x1_y1 + dist_coeffs[3] * (r2 + 2 * x1_2)
    y2 = y1 * r_dist + 2 * dist_coeffs[3] * x1_y1 + dist_coeffs[2] * (r2 + 2 * y1_2)
    # denormalize for projection (which is a linear operation)
    cam_pts_dist = torch.stack([x2 * z, y2 * z, z], dim=2).float()
    return cam_pts_dist


def rigid_tf_torch_batch(points, R, T):
    """
    Performs rigid transformation to incoming points but batched
    Q = (points*R.T) + T
    points: (batch, num, 3)
    R: (batch, 3, 3)
    T: (batch, 3, 1)
    out: (batch, num, 3)
    """
    points_out = torch.bmm(R, points.permute(0, 2, 1)) + T
    points_out = points_out.permute(0, 2, 1)
    return points_out


def solve_rigid_tf_np(A: np.ndarray, B: np.ndarray):
    """
    “Least-Squares Fitting of Two 3-D Point Sets”, Arun, K. S. , May 1987
    Input: expects Nx3 matrix of points
    Returns R,t
    R = 3x3 rotation matrix
    t = 3x1 column vector

    This function should be a fix for compute_rigid_tf when the det == -1
    """

    assert A.shape == B.shape
    A = A.T
    B = B.T

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def batch_solve_rigid_tf(A, B):
    """
    “Least-Squares Fitting of Two 3-D Point Sets”, Arun, K. S. , May 1987
    Input: expects BxNx3 matrix of points
    Returns R,t
    R = Bx3x3 rotation matrix
    t = Bx3x1 column vector
    """

    assert A.shape == B.shape
    dev = A.device
    A = A.cpu().numpy()
    B = B.cpu().numpy()
    A = permute_np(A, (0, 2, 1))
    B = permute_np(B, (0, 2, 1))

    batch, num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    _, num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=2)
    centroid_B = np.mean(B, axis=2)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(batch, -1, 1)
    centroid_B = centroid_B.reshape(batch, -1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = np.matmul(Am, permute_np(Bm, (0, 2, 1)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = np.matmul(permute_np(Vt, (0, 2, 1)), permute_np(U, (0, 2, 1)))

    # special reflection case
    neg_idx = np.linalg.det(R) < 0
    if neg_idx.sum() > 0:
        raise Exception(
            f"some rotation matrices are not orthogonal; make sure implementation is correct for such case: {neg_idx}"
        )
    Vt[neg_idx, 2, :] *= -1
    R[neg_idx, :, :] = np.matmul(
        permute_np(Vt[neg_idx], (0, 2, 1)), permute_np(U[neg_idx], (0, 2, 1))
    )

    t = np.matmul(-R, centroid_A) + centroid_B

    R = torch.FloatTensor(R).to(dev)
    t = torch.FloatTensor(t).to(dev)
    return R, t


def rigid_tf_np(points, R, T):
    """
    Performs rigid transformation to incoming points
    Q = (points*R.T) + T
    points: (num, 3)
    R: (3, 3)
    T: (1, 3)

    out: (num, 3)
    """

    assert isinstance(points, np.ndarray)
    assert isinstance(R, np.ndarray)
    assert isinstance(T, np.ndarray)
    assert len(points.shape) == 2
    assert points.shape[1] == 3
    assert R.shape == (3, 3)
    assert T.shape == (1, 3)
    points_new = np.matmul(R, points.T).T + T
    return points_new


def transform_points(world2cam_mat, pts):
    """
    Map points from one coord to another based on the 4x4 matrix.
    e.g., map points from world to camera coord.
    pts: (N, 3), in METERS!!
    world2cam_mat: (4, 4)
    Output: points in cam coord (N, 3)
    We follow this convention:
    | R T |   |pt|
    | 0 1 | * | 1|
    i.e. we rotate first then translate as T is the camera translation not position.
    """
    assert isinstance(pts, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert isinstance(world2cam_mat, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert world2cam_mat.shape == (4, 4)
    assert len(pts.shape) == 2
    assert pts.shape[1] == 3
    pts_homo = to_homo(pts)

    # mocap to cam
    pts_cam_homo = torch.matmul(world2cam_mat, pts_homo.T).T
    pts_cam = to_xyz(pts_cam_homo)

    assert pts_cam.shape[1] == 3
    return pts_cam


def transform_points_batch(world2cam_mat, pts):
    """
    Map points from one coord to another based on the 4x4 matrix.
    e.g., map points from world to camera coord.
    pts: (B, N, 3), in METERS!!
    world2cam_mat: (B, 4, 4)
    Output: points in cam coord (B, N, 3)
    We follow this convention:
    | R T |   |pt|
    | 0 1 | * | 1|
    i.e. we rotate first then translate as T is the camera translation not position.
    """
    assert isinstance(pts, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert isinstance(world2cam_mat, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert world2cam_mat.shape[1:] == (4, 4)
    assert len(pts.shape) == 3
    assert pts.shape[2] == 3
    batch_size = pts.shape[0]
    pts_homo = to_homo_batch(pts)

    # mocap to cam
    pts_cam_homo = torch.bmm(world2cam_mat, pts_homo.permute(0, 2, 1)).permute(0, 2, 1)
    pts_cam = to_xyz_batch(pts_cam_homo)

    assert pts_cam.shape[2] == 3
    return pts_cam


def project2d_batch(K, pts_cam):
    """
    K: (B, 3, 3)
    pts_cam: (B, N, 3)
    """

    assert isinstance(K, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert isinstance(pts_cam, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert K.shape[1:] == (3, 3)
    assert pts_cam.shape[2] == 3
    assert len(pts_cam.shape) == 3
    pts2d_homo = torch.bmm(K, pts_cam.permute(0, 2, 1)).permute(0, 2, 1)
    pts2d = to_xy_batch(pts2d_homo)
    return pts2d


def project2d_norm_batch(K, pts_cam, patch_width):
    """
    K: (B, 3, 3)
    pts_cam: (B, N, 3)
    """

    assert isinstance(K, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert isinstance(pts_cam, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert K.shape[1:] == (3, 3)
    assert pts_cam.shape[2] == 3
    assert len(pts_cam.shape) == 3
    v2d = project2d_batch(K, pts_cam)
    v2d_norm = data_utils.normalize_kp2d(v2d, patch_width)
    return v2d_norm


def project2d(K, pts_cam):
    assert isinstance(K, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert isinstance(pts_cam, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert K.shape == (3, 3)
    assert pts_cam.shape[1] == 3
    assert len(pts_cam.shape) == 2
    pts2d_homo = torch.matmul(K, pts_cam.T).T
    pts2d = to_xy(pts2d_homo)
    return pts2d
