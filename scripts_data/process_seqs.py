import argparse
import json
import sys
import time
import traceback
from glob import glob

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

sys.path = ["."] + sys.path
from common.body_models import construct_layers

# from src.arctic.models.object_tensors import ObjectTensors
from common.object_tensors import ObjectTensors
from src.arctic.processing import process_seq


def construct_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_verts", action="store_true")
    parser.add_argument("--mano_p", type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    dev = "cuda:0"
    args = construct_args()

    with open(
        f"./data/arctic_data/data/meta/misc.json",
        "r",
    ) as f:
        misc = json.load(f)

    statcams = {}
    for sub in misc.keys():
        statcams[sub] = {
            "world2cam": torch.FloatTensor(np.array(misc[sub]["world2cam"])),
            "intris_mat": torch.FloatTensor(np.array(misc[sub]["intris_mat"])),
        }

    if args.mano_p is not None:
        mano_ps = [args.mano_p]
    else:
        mano_ps = glob(f"./data/arctic_data/data/raw_seqs/*/*.mano.npy")

    layers = construct_layers(dev)
    # object_tensor = ObjectTensors('', './arctic_data/data')
    object_tensor = ObjectTensors()
    object_tensor.to(dev)
    layers["object"] = object_tensor

    pbar = tqdm(mano_ps)
    for mano_p in pbar:
        pbar.set_description("Processing %s" % mano_p)
        try:
            task = [mano_p, dev, statcams, layers, pbar]
            process_seq(task, export_verts=args.export_verts)
        except Exception as e:
            logger.info(traceback.format_exc())
            time.sleep(2)
            logger.info(f"Failed at {mano_p}")


if __name__ == "__main__":
    main()
