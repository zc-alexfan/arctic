import torch
from loguru import logger

import common.torch_utils as torch_utils
from common.xdict import xdict
from src.callbacks.loss.loss_arctic_lstm import compute_loss
from src.callbacks.process.process_arctic import process_data
from src.callbacks.vis.visualize_arctic import visualize_all
from src.models.arctic_lstm.model import ArcticLSTM
from src.models.generic.wrapper import GenericWrapper


class ArcticLSTMWrapper(GenericWrapper):
    def __init__(self, args):
        super().__init__(args)
        self.model = ArcticLSTM(
            focal_length=args.focal_length,
            img_res=args.img_res,
            args=args,
        )
        self.process_fn = process_data
        self.loss_fn = compute_loss
        self.metric_dict = [
            "cdev",
            "mrrpe",
            "mpjpe.ra",
            "aae",
            "success_rate",
        ]

        self.vis_fns = [visualize_all]
        self.num_vis_train = 0
        self.num_vis_val = 1

    def set_training_flags(self):
        if not self.started_training:
            sd_p = f"./logs/{self.args.img_feat_version}/checkpoints/last.ckpt"
            sd = torch.load(sd_p)["state_dict"]
            msd = xdict(sd).search("model.head")

            wd = msd.search("weight")
            bd = msd.search("bias")
            wd.merge(bd)
            self.load_state_dict(wd, strict=False)
            torch_utils.toggle_parameters(self, True)
            logger.info(f"Loaded: {sd_p}")
        self.started_training = True

    def inference(self, inputs, meta_info):
        return super().inference_pose(inputs, meta_info)
