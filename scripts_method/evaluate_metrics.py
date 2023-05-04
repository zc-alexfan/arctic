import json
import os
import sys

import torch

sys.path = ["."] + sys.path
import argparse
import os.path as op

import numpy as np
from easydict import EasyDict
from loguru import logger
from tqdm import tqdm

import common.thing as thing
from common.ld_utils import cat_dl, ld2dl
from common.xdict import xdict
from src.extraction.interface import prepare_data
from src.utils.eval_modules import eval_fn_dict


def evalute_results(
    layers, split, exp_key, setup, device, metrics, data_keys, task, eval_p
):
    with open(f"./data/arctic_data/data/splits_json/protocol_{setup}.json", "r") as f:
        protocols = json.load(f)

    seqs_val = protocols[split]

    if setup in ["p1"]:
        views = [1, 2, 3, 4, 5, 6, 7, 8]
    elif setup in ["p2"]:
        views = [0]
    else:
        assert False

    with torch.no_grad():
        all_metrics = {}
        pbar = tqdm(seqs_val)
        for seq_val in pbar:
            for view in views:
                curr_seq = seq_val.replace("/", "_") + f"_{view}"
                pbar.set_description(f"Processing {curr_seq}: load data")
                data = prepare_data(
                    curr_seq, exp_key, data_keys, layers, device, task, eval_p
                )
                pred = data.search("pred.", replace_to="")
                targets = data.search("targets.", replace_to="")
                meta_info = data.search("meta_info.", replace_to="")
                metric_dict = xdict()
                for metric in metrics:
                    pbar.set_description(f"Processing {curr_seq}: {metric}")
                    # each metric returns a tensor with shape (N, )
                    out = eval_fn_dict[metric](pred, targets, meta_info)
                    metric_dict.merge(out)
                metric_dict = metric_dict.to_np()
                all_metrics[curr_seq] = metric_dict

    agg_metrics = cat_dl(ld2dl(list(all_metrics.values())), dim=0)
    for key, val in agg_metrics.items():
        agg_metrics[key] = float(np.nanmean(thing.thing2np(val)))

    out_folder = eval_p.replace("/eval", "/results")
    if not op.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)
    np.save(op.join(out_folder, f"all_metrics_{split}_{setup}.npy"), all_metrics)
    with open(op.join(out_folder, f"agg_metrics_{split}_{setup}.json"), "w") as f:
        json.dump(agg_metrics, f, indent=4)
    logger.info(f"Exported results to {out_folder}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--eval_p", type=str, default="")
    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--setup", type=str, default="")
    config = parser.parse_args()
    args = EasyDict(vars(config))
    return args


def main():
    args = parse_args()
    from common.body_models import build_layers

    device = "cuda"
    layers = build_layers(device)

    eval_p = args.eval_p
    exp_key = eval_p.split("/")[-2]
    split = args.split
    setup = args.setup

    if "pose" in args.task:
        from src.extraction.keys.eval_pose import KEYS

        metrics = [
            "aae",
            "mpjpe.ra",
            "mrrpe",
            "success_rate",
            "cdev",
            "mdev",
            "acc_err_pose",
        ]
    elif "field" in args.task:
        from src.extraction.keys.eval_field import KEYS

        metrics = ["avg_err_field", "acc_err_field"]
    else:
        assert False

    logger.info(f"Evaluating {exp_key} {split} on setup {setup}")
    evalute_results(
        layers, split, exp_key, setup, device, metrics, KEYS, args.task, eval_p
    )


if __name__ == "__main__":
    main()
