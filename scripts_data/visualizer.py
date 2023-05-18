import argparse
import json
import os.path as op
import random
import sys
from glob import glob

import torch
from easydict import EasyDict
from loguru import logger

sys.path = ["."] + sys.path

from common.body_models import construct_layers
from common.viewer import ARCTICViewer


class DataViewer(ARCTICViewer):
    def __init__(
        self,
        render_types=["rgb", "depth", "mask"],
        interactive=True,
        size=(2024, 2024),
    ):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = construct_layers(dev)
        super().__init__(render_types, interactive, size)

    def load_data(
        self,
        seq_p,
        use_mano,
        use_object,
        use_smplx,
        no_image,
        use_distort,
        view_idx,
        subject_meta,
    ):
        logger.info("Creating meshes")
        from src.mesh_loaders.arctic import construct_meshes

        batch = construct_meshes(
            seq_p,
            self.layers,
            use_mano,
            use_object,
            use_smplx,
            no_image,
            use_distort,
            view_idx,
            subject_meta,
        )
        self.check_format(batch)
        logger.info("Done")
        return batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view_idx", type=int, default=1)
    parser.add_argument("--seq_p", type=str, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--mano", action="store_true")
    parser.add_argument("--smplx", action="store_true")
    parser.add_argument("--object", action="store_true")
    parser.add_argument("--no_image", action="store_true")
    parser.add_argument("--distort", action="store_true")
    config = parser.parse_args()
    args = EasyDict(vars(config))
    return args


def main():
    with open(
        f"./data/arctic_data/data/meta/misc.json",
        "r",
    ) as f:
        subject_meta = json.load(f)

    args = parse_args()
    random.seed(1)

    viewer = DataViewer(interactive=not args.headless, size=(2024, 2024))
    if args.seq_p is None:
        seq_ps = glob("./outputs/processed_verts/seqs/*/*.npy")
    else:
        seq_ps = [args.seq_p]
    assert len(seq_ps) > 0, f"No seqs found on {args.seq_p}"

    for seq_idx, seq_p in enumerate(seq_ps):
        logger.info(f"Rendering seq#{seq_idx+1}, seq: {seq_p}, view: {args.view_idx}")
        seq_name = seq_p.split("/")[-1].split(".")[0]
        sid = seq_p.split("/")[-2]
        out_name = f"{sid}_{seq_name}_{args.view_idx}"
        batch = viewer.load_data(
            seq_p,
            args.mano,
            args.object,
            args.smplx,
            args.no_image,
            args.distort,
            args.view_idx,
            subject_meta,
        )
        viewer.render_seq(batch, out_folder=op.join("render_out", out_name))


if __name__ == "__main__":
    main()
