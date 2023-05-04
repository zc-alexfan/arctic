import torch
from pytorch3d.ops import knn_points


def compute_dist_mano_to_obj(batch_mano_v, batch_v, batch_v_len, dist_min, dist_max):
    knn_dists, knn_idx, _ = knn_points(
        batch_mano_v, batch_v, None, batch_v_len, K=1, return_nn=True
    )
    knn_dists = knn_dists.sqrt()[:, :, 0]

    knn_dists = torch.clamp(knn_dists, dist_min, dist_max)
    return knn_dists, knn_idx[:, :, 0]


def compute_dist_obj_to_mano(batch_mano_v, batch_v, batch_v_len, dist_min, dist_max):
    knn_dists, knn_idx, _ = knn_points(
        batch_v, batch_mano_v, batch_v_len, None, K=1, return_nn=True
    )

    knn_dists = knn_dists.sqrt()
    knn_dists = torch.clamp(knn_dists, dist_min, dist_max)
    return knn_dists[:, :, 0], knn_idx[:, :, 0]


def dist2contact(dist, contact_bnd):
    contact = (dist < contact_bnd).long()
    return contact
