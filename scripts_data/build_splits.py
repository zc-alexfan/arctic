import argparse
import sys

sys.path = ["."] + sys.path
from src.arctic.split import build_split


def construct_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "all"],
        default=None,
    )
    parser.add_argument(
        "--request_keys",
        type=str,
        default="cam_coord.2d.bbox.params",
        help="save data with these keys (separated by .)",
    )
    parser.add_argument(
        "--process_folder", type=str, default="./outputs/processed/seqs"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = construct_args()
    protocol = args.protocol
    split = args.split
    request_keys = args.request_keys.split(".")
    if protocol == "all":
        protocols = [
            "p1",  # allocentric
            "p2",  # egocentric
        ]
    else:
        protocols = [protocol]

    if split == "all":
        if protocol in ["p1", "p2"]:
            splits = ["train", "val", "test"]
        else:
            raise ValueError("Unknown protocol for option 'all'")
    else:
        splits = [split]

    for protocol in protocols:
        for split in splits:
            if protocol in ["p1", "p2"]:
                assert split not in ["test"], "val/test are hidden"
            build_split(protocol, split, request_keys, args.process_folder)
