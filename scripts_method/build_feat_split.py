import argparse
import json
import os
import os.path as op
from glob import glob

import numpy as np
import torch
from easydict import EasyDict
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_p", type=str, default="")
    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--protocol", type=str, default="")
    config = parser.parse_args()
    args = EasyDict(vars(config))
    return args


def check_imgname_match(imgnames_feat, setup, split):
    print("Verifying")
    imgnames_feat = ["/".join(imgname.split("/")[-4:]) for imgname in imgnames_feat]
    data = np.load(
        op.join(f"data/arctic_data/data/splits/{setup}_{split}.npy"), allow_pickle=True
    ).item()
    imgnames = data["imgnames"]
    imgnames_npy = ["/".join(imgname.split("/")[-4:]) for imgname in imgnames]
    assert set(imgnames_npy) == set(imgnames_feat)
    print("Passed verifcation")


def main(split, protocol, eval_p):
    if protocol in ["p1"]:
        views = [1, 2, 3, 4, 5, 6, 7, 8]
    elif protocol in ["p2"]:
        views = [0]
    else:
        assert False, "Undefined protocol"

    short_split = split.replace("mini", "").replace("tiny", "")
    exp_key = eval_p.split("/")[-2]

    load_ps = glob(op.join(eval_p, "*"))
    with open(
        f"./data/arctic_data/data/splits_json/protocol_{protocol}.json", "r"
    ) as f:
        seq_names = json.load(f)[short_split]

    # needed seq/view pairs
    seq_view_specs = []
    for seq_name in seq_names:
        for view_idx in views:
            seq_view_specs.append(f"{seq_name}/{view_idx}")
    seq_view_specs = set(seq_view_specs)

    if "mini" in split:
        import random

        random.seed(1)
        random.shuffle(seq_names)
        seq_names = seq_names[:10]

    if "tiny" in split:
        import random

        random.seed(1)
        random.shuffle(seq_names)
        seq_names = seq_names[:20]

    # filter seqs within split
    _load_ps = []
    for load_p in load_ps:
        curr_seq = list(op.basename(load_p))
        view_id = int(curr_seq[-1])
        curr_seq[3] = "/"
        curr_seq = "".join(curr_seq)[:-2]  # rm view id
        if curr_seq in seq_names and view_id in views:
            _load_ps.append(load_p)

    load_ps = _load_ps
    assert len(load_ps) == len(set(load_ps))

    assert len(load_ps) > 0
    print("Loading image feat")
    vecs_list = []
    imgnames_list = []
    for load_p in tqdm(load_ps):
        feat_vec = torch.load(op.join(load_p, "preds", "pred.feat_vec.pt"))
        imgnames = torch.load(op.join(load_p, "meta_info", "meta_info.imgname.pt"))
        vecs_list.append(feat_vec)
        imgnames_list.append(imgnames)
    vecs_list = torch.cat(vecs_list, dim=0)
    imgnames_list = sum(imgnames_list, [])

    if short_split == split:
        check_imgname_match(imgnames_list, protocol, split)

    out = {"imgnames": imgnames_list, "feat_vec": vecs_list}
    out_folder = "./data/arctic_data/data/feat"
    out_p = op.join(out_folder, exp_key, f"{protocol}_{split}.pt")
    assert not op.exists(out_p), f"{out_p} already exists"
    os.makedirs(op.dirname(out_p), exist_ok=True)
    print(f"Dumping into {out_p}")
    torch.save(out, out_p)


if __name__ == "__main__":
    args = parse_args()
    split = args.split
    if split in ["all"]:
        splits = ["minitrain", "minival", "tinytest", "tinyval", "train", "val", "test"]
    else:
        splits = [split]

    for split in splits:
        print(f"Processing {split}")
        main(split, args.protocol, args.eval_p)
