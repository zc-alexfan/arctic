import torch
from loguru import logger

import common.torch_utils as torch_utils
from common.xdict import xdict
from src.callbacks.loss.loss_field import compute_loss
from src.callbacks.process.process_field import process_data
from src.callbacks.vis.visualize_field import visualize_all
from src.models.field_lstm.model import FieldLSTM
from src.models.generic.wrapper import GenericWrapper


class FieldLSTMWrapper(GenericWrapper):
    def __init__(self, args):
        super().__init__(args)
        self.model = FieldLSTM(
            "resnet50",
            args.focal_length,
            args.img_res,
            args.window_size,
        )
        self.process_fn = process_data
        self.loss_fn = compute_loss
        self.metric_dict = ["avg_err_field"]

        self.vis_fns = [visualize_all]
        self.num_vis_train = 0
        self.num_vis_val = 1

    def set_training_flags(self):
        if not self.started_training:
            sd_p = f"./logs/{self.args.img_feat_version}/checkpoints/last.ckpt"
            sd = torch.load(sd_p)["state_dict"]
            msd = xdict(sd).search("model.").rm("model.backbone")

            wd = msd.search("weight")
            bd = msd.search("bias")
            wd.merge(bd)
            self.load_state_dict(wd, strict=False)
            torch_utils.toggle_parameters(self, True)
            logger.info(f"Loaded: {sd_p}")
        self.started_training = True

    def inference(self, inputs, meta_info):
        return super().inference_field(inputs, meta_info)
