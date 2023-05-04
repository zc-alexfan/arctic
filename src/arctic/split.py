import json
import os
import os.path as op

import numpy as np
from loguru import logger
from tqdm import tqdm

# view 0 is the egocentric view
_VIEWS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
_SUBJECTS = [
    "s01",  # F
    "s02",  # F
    "s03",  # M
    "s04",  # M
    "s05",  # F
    "s06",  # M
    "s07",  # M
    "s08",  # F
    "s09",  # F
    "s10",  # M
]


def get_selected_seqs(setup, split):
    assert split in ["train", "val", "test"]

    # load seq names from json
    with open(
        op.join("./data/arctic_data/data/splits_json/", f"protocol_{setup}.json"), "r"
    ) as f:
        splits = json.load(f)

    train_seqs = splits["train"]
    val_seqs = splits["val"]
    test_seqs = splits["test"]

    # sanity check no overlap seqs
    all_seqs = train_seqs + val_seqs + test_seqs
    val_test_seqs = val_seqs + test_seqs
    assert len(set(val_test_seqs)) == len(set(val_seqs)) + len(set(test_seqs))
    for seq in val_test_seqs:
        if seq not in all_seqs:
            logger.info(seq)
            assert False, f"{seq} not in all_seqs"

    train_seqs = [seq for seq in all_seqs if seq not in val_test_seqs]
    all_seqs = train_seqs + val_test_seqs
    assert len(all_seqs) == len(set(all_seqs))

    # return
    if split == "train":
        return train_seqs
    if split == "val":
        return val_seqs
    if split == "test":
        return test_seqs


def get_selected_views(setup, split):
    # return view ids to use based on setup and split
    assert split in ["train", "val", "test"]
    assert setup in [
        "p1",
        "p2",
        "p1a",
        "p2a",
    ]
    # only static views
    if setup in ["p1", "p1a"]:
        return _VIEWS[1:]

    # seen ego view
    if setup in ["p2", "p2a"]:
        return _VIEWS[:1]


def glob_fnames(num_frames, seq, chosen_views):
    # construct paths to images
    sid, seq_name = seq.split("/")
    folder_p = op.join(f"./data/arctic_data/data/images/{sid}/{seq_name}/")

    # ignore first 10 and last 10 frames as images may be entirely black
    glob_ps = [
        op.join(folder_p, "2", "%05d.jpg" % (frame_idx))
        for frame_idx in range(10, num_frames - 10)
    ]

    # create jpg paths based on selected views
    fnames = []
    for glob_p in glob_ps:
        for view in chosen_views:
            new_p = glob_p.replace("/2/", f"/{view}/")
            fnames.append(new_p)

    assert len(fnames) == len(chosen_views) * len(glob_ps)
    assert len(fnames) == len(set(fnames))
    return fnames


def sanity_check_splits(protocol):
    # make sure no overlapping seq
    train_seqs = get_selected_seqs(protocol, "train")
    val_seqs = get_selected_seqs(protocol, "val")
    test_seqs = get_selected_seqs(protocol, "test")
    all_seqs = list(set(train_seqs + val_seqs + test_seqs))
    assert len(train_seqs) == len(set(train_seqs))
    assert len(val_seqs) == len(set(val_seqs))
    assert len(test_seqs) == len(set(test_seqs))

    train_seqs = set(train_seqs)
    val_seqs = set(val_seqs)
    test_seqs = set(test_seqs)
    assert len(set.intersection(train_seqs, val_seqs)) == 0
    assert len(set.intersection(train_seqs, test_seqs)) == 0
    assert len(set.intersection(test_seqs, val_seqs)) == 0
    assert len(all_seqs) == len(train_seqs) + len(val_seqs) + len(test_seqs)


def sanity_check_annot(seq_name, data):
    # make sure no NaN or Inf
    num_frames = data["params"]["pose_r"].shape[0]
    for pkey, side_dict in data.items():
        if isinstance(side_dict, dict):
            for key, val in side_dict.items():
                if "smplx" in key:
                    # smplx distortion can be undefined
                    continue
                assert np.isnan(val).sum() == 0, f"{seq_name}: {pkey}_{key} has NaN"
                assert np.isinf(val).sum() == 0, f"{seq_name}: {pkey}_{key} has Inf"
                assert num_frames == val.shape[0]
        else:
            if "smplx" in pkey:
                # smplx distortion can be undefined
                continue
            assert np.isnan(side_dict).sum() == 0, f"{seq_name}: {pkey}_{key} has NaN"
            assert np.isinf(side_dict).sum() == 0, f"{seq_name}: {pkey}_{key} has Inf"
            assert num_frames == side_dict.shape[0]


def build_split(protocol, split, request_keys, process_folder):
    logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    logger.info(f"Constructing split {split} for protocol {protocol}")
    # extract seq_names
    # unpack protocol
    sanity_check_splits(protocol)
    chosen_seqs = get_selected_seqs(protocol, split)
    logger.info(f"Chosen {len(chosen_seqs)} seqs:")
    logger.info(chosen_seqs)
    chosen_views = get_selected_views(protocol, split)
    logger.info(f"Chosen {len(chosen_views)} views:")
    logger.info(chosen_views)
    fseqs = chosen_seqs

    # do not need world in reconstruction
    data_dict = {}
    for seq in tqdm(fseqs):
        seq_p = op.join(process_folder, f"{seq}.npy")
        if "_verts" in seq_p:
            logger.warning(
                "Trying to build split with verts. This will require lots of storage"
            )
        data = np.load(seq_p, allow_pickle=True).item()
        sanity_check_annot(seq_p, data)
        data = {k: v for k, v in data.items() if k in request_keys}
        data_dict[seq] = data

    logger.info(f"Constructing image filenames from {len(fseqs)} seqs")
    fnames = []
    for seq in tqdm(fseqs):
        fnames.append(
            glob_fnames(data_dict[seq]["params"]["rot_r"].shape[0], seq, chosen_views)
        )
    fnames = sum(fnames, [])
    assert len(fnames) == len(set(fnames))

    logger.info(f"Done. Total {len(fnames)} images")

    out_data = {}
    out_data["data_dict"] = data_dict
    out_data["imgnames"] = fnames

    if "_verts" in process_folder:
        out_p = f"./outputs/splits_verts/{protocol}_{split}.npy"
    else:
        out_p = f"./outputs/splits/{protocol}_{split}.npy"
    out_folder = op.dirname(out_p)
    if not op.exists(out_folder):
        os.makedirs(out_folder)
    logger.info("Dumping data")
    np.save(out_p, out_data)
    logger.info(f"Exported: {out_p}")
