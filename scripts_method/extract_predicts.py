import os
import json
import os.path as op
import sys
from pprint import pformat

import torch
from loguru import logger
from tqdm import tqdm

sys.path.append(".")
import common.thing as thing
import src.extraction.interface as interface
import src.factory as factory
from common.xdict import xdict
from src.parsers.parser import construct_args


# LSTM models are trained using image features from single-frame models
# this specify the single-frame model features that the LSTM model was trained on
# model_dependencies[lstm_model_id] = single_frame_model_id
model_dependencies = {
    "423c6057b": "3558f1342",
    "40ae50712": "28bf3642f",
    "546c1e997": "1f9ac0b15",
    "701a72569": "58e200d16",
    "fdc34e6c3": "66417ff6e",
    "49abdaee9": "7d09884c6",
    "5e6f6aeb9": "fb59bac27",
    "ec90691f8": "782c39821",
}


def main():
    args = construct_args()

    args.experiment = None
    args.exp_key = "xxxxxxx"

    device = "cuda:0"
    wrapper = factory.fetch_model(args).to(device)
    assert args.load_ckpt != ""
    wrapper.load_state_dict(torch.load(args.load_ckpt)["state_dict"])
    logger.info(f"Loaded weights from {args.load_ckpt}")
    wrapper.eval()
    wrapper.to(device)
    wrapper.model.arti_head.object_tensors.to(device)
    # wrapper.metric_dict = []

    exp_key = op.abspath(args.load_ckpt).split("/")[-3]
    if exp_key in model_dependencies.keys():
        assert (
            args.img_feat_version == model_dependencies[exp_key]
        ), f"Image features used for training ({model_dependencies[exp_key]}) do not match the ones used for the current inference ({args.img_feat_version})"

    out_dir = op.join(args.load_ckpt.split("checkpoints")[0], "eval")

    with open(
        f"./data/arctic_data/data/splits_json/protocol_{args.setup}.json",
        "r",
    ) as f:
        seqs = json.load(f)[args.run_on]

    logger.info(f"Hyperparameters: \n {pformat(args)}")
    logger.info(f"Seqs to process ({len(seqs)}): {seqs}")

    if args.extraction_mode in ["eval_pose"]:
        from src.extraction.keys.eval_pose import KEYS
    elif args.extraction_mode in ["eval_field"]:
        from src.extraction.keys.eval_field import KEYS
    elif args.extraction_mode in ["submit_pose"]:
        from src.extraction.keys.submit_pose import KEYS
    elif args.extraction_mode in ["submit_field"]:
        from src.extraction.keys.submit_field import KEYS
    elif args.extraction_mode in ["feat_pose"]:
        from src.extraction.keys.feat_pose import KEYS
    elif args.extraction_mode in ["feat_field"]:
        from src.extraction.keys.feat_field import KEYS
    elif args.extraction_mode in ["vis_pose"]:
        from src.extraction.keys.vis_pose import KEYS
    elif args.extraction_mode in ["vis_field"]:
        from src.extraction.keys.vis_field import KEYS
    else:
        assert False, f"Invalid extract ({args.extraction_mode})"

    if "submit_" in args.extraction_mode:
        task = args.extraction_mode.replace('submit_', '')
        task_name = f'{task}_{args.setup}_test'
        out_dir = out_dir.replace('/eval', f'/submit/{task_name}/eval')
        os.makedirs(out_dir, exist_ok=True)

    for seq_idx, seq in enumerate(seqs):
        logger.info(f"Processing seq {seq} {seq_idx + 1}/{len(seqs)}")
        out_list = []
        val_loader = factory.fetch_dataloader(args, "val", seq)
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
                batch = thing.thing2dev(batch, device)
                inputs, targets, meta_info = batch
                if "submit_" in args.extraction_mode:
                    out_dict = wrapper.inference(inputs, meta_info)
                else:
                    out_dict = wrapper.forward(inputs, targets, meta_info, "extract")
                out_dict = xdict(out_dict)
                out_dict = out_dict.subset(KEYS)
                out_list.append(out_dict)

        out = interface.std_interface(out_list)
        interface.save_results(out, out_dir)
        logger.info("Done")

    if 'submit_' in args.extraction_mode:
        import shutil
        zip_name = f'{task_name}'
        zip_path = op.join(out_dir, zip_name).replace(f'/submit/{task_name}/eval/', '/submit/')
        shutil.make_archive(zip_path, 'zip', root_dir=op.dirname(zip_path), base_dir=op.basename(zip_path))
        logger.info(f"Your submission file as exported at {zip_path}.zip")


if __name__ == "__main__":
    main()
