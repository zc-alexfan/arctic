import numpy as np
import torch

"""
Useful geometric operations, e.g. Perspective projection and a differentiable Rodrigues formula
Parts of the code are taken from https://github.com/MandyMo/pytorch_HMR
"""


def perspective_to_weak_perspective_torch(
    perspective_camera,
    focal_length,
    img_res,
):
    # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz]
    # in 3D given the bounding box size
    # This camera translation can be used in a full perspective projection
    # if isinstance(focal_length, torch.Tensor):
    #     focal_length = focal_length[:, 0]

    tx = perspective_camera[:, 0]
    ty = perspective_camera[:, 1]
    tz = perspective_camera[:, 2]

    weak_perspective_camera = torch.stack(
        [2 * focal_length / (img_res * tz + 1e-9), tx, ty],
        dim=-1,
    )
    return weak_perspective_camera


def convert_perspective_to_weak_perspective(
    perspective_camera,
    focal_length,
    img_res,
):
    # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz]
    # in 3D given the bounding box size
    # This camera translation can be used in a full perspective projection
    # if isinstance(focal_length, torch.Tensor):
    #     focal_length = focal_length[:, 0]

    weak_perspective_camera = torch.stack(
        [
            2 * focal_length / (img_res * perspective_camera[:, 2] + 1e-9),
            perspective_camera[:, 0],
            perspective_camera[:, 1],
        ],
        dim=-1,
    )
    return weak_perspective_camera


def convert_weak_perspective_to_perspective(
    weak_perspective_camera, focal_length, img_res
):
    # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz]
    # in 3D given the bounding box size
    # This camera translation can be used in a full perspective projection
    # if isinstance(focal_length, torch.Tensor):
    #     focal_length = focal_length[:, 0]

    perspective_camera = torch.stack(
        [
            weak_perspective_camera[:, 1],
            weak_perspective_camera[:, 2],
            2 * focal_length / (img_res * weak_perspective_camera[:, 0] + 1e-9),
        ],
        dim=-1,
    )
    return perspective_camera


def get_default_cam_t(f, img_res):
    cam = torch.tensor([[5.0, 0.0, 0.0]])
    return convert_weak_perspective_to_perspective(cam, f, img_res)


def estimate_translation_np(S, joints_2d, joints_conf, focal_length, img_size):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """
    num_joints = S.shape[0]
    # focal length

    f = np.array([focal_length[0], focal_length[1]])
    # optical center
    center = np.array([img_size[1] / 2.0, img_size[0] / 2.0])

    # transformations
    Z = np.reshape(np.tile(S[:, 2], (2, 1)).T, -1)
    XY = np.reshape(S[:, 0:2], -1)
    O = np.tile(center, num_joints)
    F = np.tile(f, num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf), (2, 1)).T, -1)

    # least squares
    Q = np.array(
        [
            F * np.tile(np.array([1, 0]), num_joints),
            F * np.tile(np.array([0, 1]), num_joints),
            O - np.reshape(joints_2d, -1),
        ]
    ).T
    c = (np.reshape(joints_2d, -1) - O) * Z - F * XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W, Q)
    c = np.dot(W, c)

    # square matrix
    A = np.dot(Q.T, Q)
    b = np.dot(Q.T, c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans


def estimate_translation(
    S,
    joints_2d,
    focal_length,
    img_size,
    use_all_joints=False,
    rotation=None,
    pad_2d=False,
):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (B, 49, 3) 3D joint locations
        joints: (B, 49, 3) 2D joint locations and confidence
    Returns:
        (B, 3) camera translation vectors
    """
    if pad_2d:
        batch, num_pts = joints_2d.shape[:2]
        joints_2d_pad = torch.ones((batch, num_pts, 3))
        joints_2d_pad[:, :, :2] = joints_2d
        joints_2d_pad = joints_2d_pad.to(joints_2d.device)
        joints_2d = joints_2d_pad

    device = S.device

    if rotation is not None:
        S = torch.einsum("bij,bkj->bki", rotation, S)

    # Use only joints 25:49 (GT joints)
    if use_all_joints:
        S = S.cpu().numpy()
        joints_2d = joints_2d.cpu().numpy()
    else:
        S = S[:, 25:, :].cpu().numpy()
        joints_2d = joints_2d[:, 25:, :].cpu().numpy()

    joints_conf = joints_2d[:, :, -1]
    joints_2d = joints_2d[:, :, :-1]
    trans = np.zeros((S.shape[0], 3), dtype=np.float32)
    # Find the translation for each example in the batch
    for i in range(S.shape[0]):
        S_i = S[i]
        joints_i = joints_2d[i]
        conf_i = joints_conf[i]
        trans[i] = estimate_translation_np(
            S_i, joints_i, conf_i, focal_length=focal_length, img_size=img_size
        )
    return torch.from_numpy(trans).to(device)


def estimate_translation_cam(
    S, joints_2d, focal_length, img_size, use_all_joints=False, rotation=None
):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (B, 49, 3) 3D joint locations
        joints: (B, 49, 3) 2D joint locations and confidence
    Returns:
        (B, 3) camera translation vectors
    """

    def estimate_translation_np(S, joints_2d, joints_conf, focal_length, img_size):
        """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
        Input:
            S: (25, 3) 3D joint locations
            joints: (25, 3) 2D joint locations and confidence
        Returns:
            (3,) camera translation vector
        """

        num_joints = S.shape[0]
        # focal length
        f = np.array([focal_length[0], focal_length[1]])
        # optical center
        center = np.array([img_size[0] / 2.0, img_size[1] / 2.0])

        # transformations
        Z = np.reshape(np.tile(S[:, 2], (2, 1)).T, -1)
        XY = np.reshape(S[:, 0:2], -1)
        O = np.tile(center, num_joints)
        F = np.tile(f, num_joints)
        weight2 = np.reshape(np.tile(np.sqrt(joints_conf), (2, 1)).T, -1)

        # least squares
        Q = np.array(
            [
                F * np.tile(np.array([1, 0]), num_joints),
                F * np.tile(np.array([0, 1]), num_joints),
                O - np.reshape(joints_2d, -1),
            ]
        ).T
        c = (np.reshape(joints_2d, -1) - O) * Z - F * XY

        # weighted least squares
        W = np.diagflat(weight2)
        Q = np.dot(W, Q)
        c = np.dot(W, c)

        # square matrix
        A = np.dot(Q.T, Q)
        b = np.dot(Q.T, c)

        # solution
        trans = np.linalg.solve(A, b)

        return trans

    device = S.device

    if rotation is not None:
        S = torch.einsum("bij,bkj->bki", rotation, S)

    # Use only joints 25:49 (GT joints)
    if use_all_joints:
        S = S.cpu().numpy()
        joints_2d = joints_2d.cpu().numpy()
    else:
        S = S[:, 25:, :].cpu().numpy()
        joints_2d = joints_2d[:, 25:, :].cpu().numpy()

    joints_conf = joints_2d[:, :, -1]
    joints_2d = joints_2d[:, :, :-1]
    trans = np.zeros((S.shape[0], 3), dtype=np.float32)
    # Find the translation for each example in the batch
    for i in range(S.shape[0]):
        S_i = S[i]
        joints_i = joints_2d[i]
        conf_i = joints_conf[i]
        trans[i] = estimate_translation_np(
            S_i, joints_i, conf_i, focal_length=focal_length, img_size=img_size
        )
    return torch.from_numpy(trans).to(device)


def get_coord_maps(size=56):
    xx_ones = torch.ones([1, size], dtype=torch.int32)
    xx_ones = xx_ones.unsqueeze(-1)

    xx_range = torch.arange(size, dtype=torch.int32).unsqueeze(0)
    xx_range = xx_range.unsqueeze(1)

    xx_channel = torch.matmul(xx_ones, xx_range)
    xx_channel = xx_channel.unsqueeze(-1)

    yy_ones = torch.ones([1, size], dtype=torch.int32)
    yy_ones = yy_ones.unsqueeze(1)

    yy_range = torch.arange(size, dtype=torch.int32).unsqueeze(0)
    yy_range = yy_range.unsqueeze(-1)

    yy_channel = torch.matmul(yy_range, yy_ones)
    yy_channel = yy_channel.unsqueeze(-1)

    xx_channel = xx_channel.permute(0, 3, 1, 2)
    yy_channel = yy_channel.permute(0, 3, 1, 2)

    xx_channel = xx_channel.float() / (size - 1)
    yy_channel = yy_channel.float() / (size - 1)

    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1

    out = torch.cat([xx_channel, yy_channel], dim=1)
    return out


def look_at(eye, at=np.array([0, 0, 0]), up=np.array([0, 0, 1]), eps=1e-5):
    at = at.astype(float).reshape(1, 3)
    up = up.astype(float).reshape(1, 3)

    eye = eye.reshape(-1, 3)
    up = up.repeat(eye.shape[0] // up.shape[0], axis=0)
    eps = np.array([eps]).reshape(1, 1).repeat(up.shape[0], axis=0)

    z_axis = eye - at
    z_axis /= np.max(np.stack([np.linalg.norm(z_axis, axis=1, keepdims=True), eps]))

    x_axis = np.cross(up, z_axis)
    x_axis /= np.max(np.stack([np.linalg.norm(x_axis, axis=1, keepdims=True), eps]))

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.max(np.stack([np.linalg.norm(y_axis, axis=1, keepdims=True), eps]))

    r_mat = np.concatenate(
        (x_axis.reshape(-1, 3, 1), y_axis.reshape(-1, 3, 1), z_axis.reshape(-1, 3, 1)),
        axis=2,
    )

    return r_mat


def to_sphere(u, v):
    theta = 2 * np.pi * u
    phi = np.arccos(1 - 2 * v)
    cx = np.sin(phi) * np.cos(theta)
    cy = np.sin(phi) * np.sin(theta)
    cz = np.cos(phi)
    s = np.stack([cx, cy, cz])
    return s


def sample_on_sphere(range_u=(0, 1), range_v=(0, 1)):
    u = np.random.uniform(*range_u)
    v = np.random.uniform(*range_v)
    return to_sphere(u, v)


def sample_pose_on_sphere(range_v=(0, 1), range_u=(0, 1), radius=1, up=[0, 1, 0]):
    # sample location on unit sphere
    loc = sample_on_sphere(range_u, range_v)

    # sample radius if necessary
    if isinstance(radius, tuple):
        radius = np.random.uniform(*radius)

    loc = loc * radius
    R = look_at(loc, up=np.array(up))[0]

    RT = np.concatenate([R, loc.reshape(3, 1)], axis=1)
    RT = torch.Tensor(RT.astype(np.float32))
    return RT


def rectify_pose(camera_r, body_aa, rotate_x=False):
    body_r = batch_rodrigues(body_aa).reshape(-1, 3, 3)

    if rotate_x:
        rotate_x = torch.tensor([[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]])
        body_r = body_r @ rotate_x

    final_r = camera_r @ body_r
    body_aa = batch_rot2aa(final_r)
    return body_aa


def estimate_translation_k_np(S, joints_2d, joints_conf, K):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """
    num_joints = S.shape[0]
    # focal length

    focal = np.array([K[0, 0], K[1, 1]])
    # optical center
    center = np.array([K[0, 2], K[1, 2]])

    # transformations
    Z = np.reshape(np.tile(S[:, 2], (2, 1)).T, -1)
    XY = np.reshape(S[:, 0:2], -1)
    O = np.tile(center, num_joints)
    F = np.tile(focal, num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf), (2, 1)).T, -1)

    # least squares
    Q = np.array(
        [
            F * np.tile(np.array([1, 0]), num_joints),
            F * np.tile(np.array([0, 1]), num_joints),
            O - np.reshape(joints_2d, -1),
        ]
    ).T
    c = (np.reshape(joints_2d, -1) - O) * Z - F * XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W, Q)
    c = np.dot(W, c)

    # square matrix
    A = np.dot(Q.T, Q)
    b = np.dot(Q.T, c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans


def estimate_translation_k(
    S,
    joints_2d,
    K,
    use_all_joints=False,
    rotation=None,
    pad_2d=False,
):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (B, 49, 3) 3D joint locations
        joints: (B, 49, 3) 2D joint locations and confidence
    Returns:
        (B, 3) camera translation vectors
    """
    if pad_2d:
        batch, num_pts = joints_2d.shape[:2]
        joints_2d_pad = torch.ones((batch, num_pts, 3))
        joints_2d_pad[:, :, :2] = joints_2d
        joints_2d_pad = joints_2d_pad.to(joints_2d.device)
        joints_2d = joints_2d_pad

    device = S.device

    if rotation is not None:
        S = torch.einsum("bij,bkj->bki", rotation, S)

    # Use only joints 25:49 (GT joints)
    if use_all_joints:
        S = S.cpu().numpy()
        joints_2d = joints_2d.cpu().numpy()
    else:
        S = S[:, 25:, :].cpu().numpy()
        joints_2d = joints_2d[:, 25:, :].cpu().numpy()

    joints_conf = joints_2d[:, :, -1]
    joints_2d = joints_2d[:, :, :-1]
    trans = np.zeros((S.shape[0], 3), dtype=np.float32)
    # Find the translation for each example in the batch
    for i in range(S.shape[0]):
        S_i = S[i]
        joints_i = joints_2d[i]
        conf_i = joints_conf[i]
        K_i = K[i]
        trans[i] = estimate_translation_k_np(S_i, joints_i, conf_i, K_i)
    return torch.from_numpy(trans).to(device)


def weak_perspective_to_perspective_torch(
    weak_perspective_camera, focal_length, img_res, min_s
):
    # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz]
    # in 3D given the bounding box size
    # This camera translation can be used in a full perspective projection
    s = weak_perspective_camera[:, 0]
    s = torch.clamp(s, min_s)
    tx = weak_perspective_camera[:, 1]
    ty = weak_perspective_camera[:, 2]
    perspective_camera = torch.stack(
        [
            tx,
            ty,
            2 * focal_length / (img_res * s + 1e-9),
        ],
        dim=-1,
    )
    return perspective_camera
