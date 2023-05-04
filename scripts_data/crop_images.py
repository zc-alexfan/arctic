import argparse
import json
import os
import os.path as op
import time
import traceback
from glob import glob

import numpy as np
from loguru import logger
from PIL import Image
from tqdm import tqdm

logger.add("file_{time}.log")


EGO_IMAGE_SCALE = 0.3

with open(
    f"./arctic_data/meta/misc.json",
    "r",
) as f:
    misc = json.load(f)


def transform_image(im, bbox_loose, cap_dim):
    cx, cy, dim = bbox_loose.copy()
    dim *= 200
    im_cropped = im.crop((cx - dim / 2, cy - dim / 2, cx + dim / 2, cy + dim / 2))

    im_cropped_cap = im_cropped.resize((cap_dim, cap_dim))
    return im_cropped_cap


def process_fname(fname, bbox_loose, sid, view_idx, pbar):
    vidx = int(op.basename(fname).split(".")[0]) - misc[sid]["ioi_offset"]
    out_p = fname.replace("./data/arctic_data/data/images", "./outputs/croppped_images")
    num_frames = bbox_loose.shape[0]

    if vidx < 0:
        # expected
        return True

    if vidx >= num_frames:
        # not expected
        return False

    if op.exists(out_p):
        return True

    pbar.set_description(f"Croppping {fname}")
    im = Image.open(fname)
    if view_idx > 0:
        im_cap = transform_image(im, bbox_loose[vidx], cap_dim=1000)
    else:
        width, height = im.size
        width_new = int(width * EGO_IMAGE_SCALE)
        height_new = int(height * EGO_IMAGE_SCALE)
        im_cap = im.resize((width_new, height_new))
    out_folder = op.dirname(out_p)
    if not op.exists(out_folder):
        os.makedirs(out_folder)

    im_cap.save(out_p)
    return True


def process_seq(seq_p):
    print(f"Start {seq_p}")

    seq_data = np.load(seq_p, allow_pickle=True).item()
    sid, seq_name = seq_p.split("/")[-2:]

    seq_name = seq_name.split(".")[0]
    stamp = time.time()

    for view_idx in range(9):
        print(f"Processing view#{view_idx}")
        bbox = seq_data["bbox"][:, view_idx]
        bbox_loose = bbox.copy()
        bbox_loose[:, 2] *= 1.5  # 1.5X around the bbox

        fnames = glob(
            f"./data/arctic_data/data/images/{sid}/{seq_name}/{view_idx}/*.jpg"
        )
        fnames = sorted(fnames)
        if len(fnames) == 0:
            logger.info(f"No images in {sid}/{seq_name}/{view_idx}")

        pbar = tqdm(fnames)
        for fname in pbar:
            try:
                status = process_fname(fname, bbox_loose, sid, view_idx, pbar)
                if status is False:
                    logger.info(f"Skip due to no GT: {fname}")
            except:
                traceback.print_exc()
                logger.info(f"Skip due to Exception: {fname}")
                time.sleep(1.0)

    print(f"Done! Elapsed {time.time() - stamp:.2f}s")


def construct_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, default=None)
    parser.add_argument(
        "--process_folder", type=str, default="./outputs/processed/seqs"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = construct_args()
    seq_ps = glob(op.join(args.process_folder, "*/*.npy"))
    seq_ps = sorted(seq_ps)
    assert len(seq_ps) > 0

    if args.task_id < 0:
        for seq_p in seq_ps:
            process_seq(seq_p)
    else:
        seq_p = seq_ps[args.task_id]
        process_seq(seq_p)
