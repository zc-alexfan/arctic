import numpy as np
import torch

import common.torch_utils as torch_utils
from common.xdict import xdict


def find_windows(dist, dist_idx, vo, contact_thres, window_thres):
    # find windows with at least `window_thres` frames in continuous contact
    # dist: closest distance of each MANO vertex to object for every frame, (num_frames, 778)
    # dist_idx: closest object vertex id of each MANO vertex to object for every frame, (num_frames, 778)
    # vo: object vertices in a static frame, (num_obj_verts, 3)
    # contact_thres: threshold for contact
    # window_thres: threshold for window length

    # return: windows tensor in shape (num_windows, 4) where each window is [m, n, i, j]
    # m: start frame
    # n: end frame
    # i: hand vertex id
    # j: object vertex id

    assert isinstance(dist, (torch.Tensor))
    assert isinstance(dist_idx, (torch.Tensor))
    num_frames, num_verts = dist.shape
    contacts = (dist < contact_thres).bool()

    # find MANO vertices that are in contact for at least `window_thres` frames (not necessarily continuous at this point)
    # the goal is to reduce number ofM MANO vertices to search
    verts_ids = (contacts.long().sum(dim=0) >= window_thres).nonzero().view(-1).tolist()
    windows = []
    # search for each potential MANO vertex id
    for vidx in verts_ids:
        window_s = None
        window_e = None
        prev_in_contact = False
        # loop along time dimension
        for fidx in range(num_frames):
            if not prev_in_contact and contacts[fidx, vidx]:
                # if prev not in contact, and current in contact
                # this indicates the start of a window
                window_s = fidx  # start of window
                prev_in_contact = True
            elif prev_in_contact and contacts[fidx, vidx]:
                # if prev in contact, and current in contact
                # inside contact window
                continue
            elif not prev_in_contact and not contacts[fidx, vidx]:
                # if prev not in contact, and current not in contact
                # gaps between contact windows
                continue
            elif prev_in_contact and not contacts[fidx, vidx]:
                # prev in contact, current not in contact
                # end of contact window
                # window found: [window_s, window_e] (inclusive)
                window_e = fidx - 1  # end of window
                prev_in_contact = False  # reset
                # skip len(window) < window_thres
                if window_e - window_s + 1 < window_thres:
                    continue

                # remove windows with sliding finger along object surface
                # check max distance of object vertices matched to hand vertex i
                # if > 3mm skip this in windows
                j_list = dist_idx[
                    window_s : window_e + 1, vidx
                ]  # object vertex ids that are closest to hand vertex vidx within window
                vj = vo[
                    j_list
                ]  # object vertices closest to hand vertex vidx within window
                cdist = (
                    torch.cdist(vj, vj).cpu().numpy()
                )  # check if they are nearby in a canonical static frame
                triu_idx = torch.triu_indices(window_thres, window_thres)
                cdist[triu_idx[0, :], triu_idx[1, :]] = float(
                    "nan"
                )  # remove upper triangle (duplicates)
                mean_dist = np.nanmean(
                    cdist.reshape(-1)
                )  # average distance between object vertices

                # mano vertex vidx has slided along object surface
                if mean_dist > contact_thres:
                    continue
                else:
                    # find the most frequent object vertex id to match hand vertex vidx
                    jidx = int(torch.mode(j_list)[0])
                windows.append([window_s, window_e, vidx, jidx])
            else:
                assert False

    # verify each window has continuous contact and is the biggest
    for window in windows:
        line = (
            contacts[window[0] - 1 : window[1] + 1 + 1, window[2] : window[2] + 1]
            .long()
            .view(-1)
        )
        # check if the window is the biggest
        assert not contacts[window[0] - 1, window[2]]
        assert not contacts[window[1] + 1, window[2]]

        # Example 011110 gives line.sum() == 4 and len(line) == 6
        assert line.sum() == len(line) - 2
    return windows


def find_windows_wrapper(dist, dist_idx, vo, contact_thres, window_thres):
    # find windows with at least `window_thres` frames in continuous contact
    windows = np.array(find_windows(dist, dist_idx, vo[0], contact_thres, window_thres))
    return windows


def compute_mdev(windows, pred_vh, pred_vo, frame_valid):
    mdev_list = []
    for window in windows:
        m, n, i, j = window
        # extract hand object locations according to pairs
        pred_stable_vh = pred_vh[m : n + 1, i]
        pred_stable_vo = pred_vo[m : n + 1, j]

        # direction of hand and object vertices in time
        pred_delta_vh = pred_stable_vh[1:] - pred_stable_vh[:-1]
        pred_delta_vo = pred_stable_vo[1:] - pred_stable_vo[:-1]

        # difference between hand and object directions
        pred_diff_delta = pred_delta_vh - pred_delta_vo

        # a diff is valid if two consecutive frames are valid
        valid = frame_valid[m : n + 1].clone()
        diff_valid = valid[1:] * valid[:-1]
        diff_valid = diff_valid.bool()

        # set invalid diff to nan
        pred_diff_delta[~diff_valid, :] = float("nan")

        mdev = torch.norm(pred_diff_delta, dim=1)

        # normalize by (valid) window size
        mdev = torch_utils.nanmean(mdev, dim=0)
        mdev_list.append(mdev)
    return mdev_list


def eval_motion_deviation(pred, targets, meta_info):
    num_frames, num_verts = pred["mano.v3d.cam.r"].shape[:2]

    is_valid = targets["is_valid"]
    r_valid = targets["right_valid"] * is_valid
    l_valid = targets["left_valid"] * is_valid

    # parameters
    contact_thres = 3e-3
    window_thres = 15  # half a second

    # find stable contact window btw right and object
    # [m, n, i, j]
    windows_r = find_windows_wrapper(
        targets["dist.ro"],
        targets["idx.ro"],
        targets["object.v.cam"],
        contact_thres,
        window_thres,
    )

    # left hand
    windows_l = find_windows_wrapper(
        targets["dist.lo"],
        targets["idx.lo"],
        targets["object.v.cam"],
        contact_thres,
        window_thres,
    )

    mdev_list_r = compute_mdev(
        windows_r, pred["mano.v3d.cam.r"], pred["object.v.cam"], r_valid
    )
    mdev_r = torch.stack(mdev_list_r)

    mdev_list_l = compute_mdev(
        windows_l, pred["mano.v3d.cam.l"], pred["object.v.cam"], l_valid
    )
    mdev_l = torch.stack(mdev_list_l)

    mdev_h = torch.cat((mdev_r, mdev_l), dim=0)

    metric_dict = xdict()
    # metric_dict["mdev/r"] = mdev_r
    # metric_dict["mdev/l"] = mdev_l
    metric_dict["mdev/h"] = mdev_h
    metric_dict = metric_dict.mul(1000).to_np()  # mm

    return metric_dict
