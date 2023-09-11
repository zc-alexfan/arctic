import numpy as np
import torch

import common.data_utils as data_utils
import common.ld_utils as ld_utils
import src.callbacks.process.process_generic as generic
from common.abstract_pl import AbstractPL
from common.body_models import MANODecimator, build_mano_aa
from common.comet_utils import push_images
from common.rend_utils import Renderer
from common.xdict import xdict
from src.utils.eval_modules import eval_fn_dict


def mul_loss_dict(loss_dict):
    for key, val in loss_dict.items():
        loss, weight = val
        loss_dict[key] = loss * weight
    return loss_dict


class GenericWrapper(AbstractPL):
    def __init__(self, args):
        super().__init__(
            args,
            push_images,
            "loss__val",
            float("inf"),
            high_loss_val=float("inf"),
        )
        self.args = args
        self.mano_r = build_mano_aa(is_rhand=True)
        self.mano_l = build_mano_aa(is_rhand=False)
        self.add_module("mano_r", self.mano_r)
        self.add_module("mano_l", self.mano_l)
        self.renderer = Renderer(img_res=args.img_res)
        self.object_sampler = np.load(
            "./data/arctic_data/data/meta/downsamplers.npy", allow_pickle=True
        ).item()

    def set_flags(self, mode):
        self.model.mode = mode
        if mode == "train":
            self.train()
        else:
            self.eval()

    def inference_pose(self, inputs, meta_info):
        pred = self.model(inputs, meta_info)
        mydict = xdict()
        mydict.merge(xdict(inputs).prefix("inputs."))
        mydict.merge(pred.prefix("pred."))
        mydict.merge(xdict(meta_info).prefix("meta_info."))
        mydict = mydict.detach()
        return mydict

    def inference_field(self, inputs, meta_info):
        meta_info = xdict(meta_info)

        models = {
            "mano_r": self.mano_r,
            "mano_l": self.mano_l,
            "arti_head": self.model.arti_head,
            "mesh_sampler": MANODecimator(),
            "object_sampler": self.object_sampler,
        }

        batch_size = meta_info["intrinsics"].shape[0]

        (
            v0_r,
            v0_l,
            v0_o,
            pidx,
            v0_r_full,
            v0_l_full,
            v0_o_full,
            mask,
            cams,
        ) = generic.prepare_templates(
            batch_size,
            models["mano_r"],
            models["mano_l"],
            models["mesh_sampler"],
            models["arti_head"],
            meta_info["query_names"],
        )

        meta_info["v0.r"] = v0_r
        meta_info["v0.l"] = v0_l
        meta_info["v0.o"] = v0_o

        pred = self.model(inputs, meta_info)
        mydict = xdict()
        mydict.merge(xdict(inputs).prefix("inputs."))
        mydict.merge(pred.prefix("pred."))
        mydict.merge(meta_info.prefix("meta_info."))
        mydict = mydict.detach()
        return mydict

    def forward(self, inputs, targets, meta_info, mode):
        models = {
            "mano_r": self.mano_r,
            "mano_l": self.mano_l,
            "arti_head": self.model.arti_head,
            "mesh_sampler": MANODecimator(),
            "object_sampler": self.object_sampler,
        }

        self.set_flags(mode)
        inputs = xdict(inputs)
        targets = xdict(targets)
        meta_info = xdict(meta_info)
        with torch.no_grad():
            inputs, targets, meta_info = self.process_fn(
                models, inputs, targets, meta_info, mode, self.args
            )

        move_keys = ["object.v_len"]
        for key in move_keys:
            meta_info[key] = targets[key]
        meta_info["mano.faces.r"] = self.mano_r.faces
        meta_info["mano.faces.l"] = self.mano_l.faces
        pred = self.model(inputs, meta_info)
        loss_dict = self.loss_fn(
            pred=pred, gt=targets, meta_info=meta_info, args=self.args
        )
        loss_dict = {k: (loss_dict[k][0].mean(), loss_dict[k][1]) for k in loss_dict}
        loss_dict = mul_loss_dict(loss_dict)
        loss_dict["loss"] = sum(loss_dict[k] for k in loss_dict)

        # conversion for vis and eval
        keys = list(pred.keys())
        for key in keys:
            # denormalize 2d keypoints
            if "2d.norm" in key:
                denorm_key = key.replace(".norm", "")
                assert key in targets.keys(), f"Do not have key {key}"

                val_pred = pred[key]
                val_gt = targets[key]

                val_denorm_pred = data_utils.unormalize_kp2d(
                    val_pred, self.args.img_res
                )
                val_denorm_gt = data_utils.unormalize_kp2d(val_gt, self.args.img_res)

                pred[denorm_key] = val_denorm_pred
                targets[denorm_key] = val_denorm_gt

        if mode == "train":
            return {"out_dict": (inputs, targets, meta_info, pred), "loss": loss_dict}

        if mode == "vis":
            vis_dict = xdict()
            vis_dict.merge(inputs.prefix("inputs."))
            vis_dict.merge(pred.prefix("pred."))
            vis_dict.merge(targets.prefix("targets."))
            vis_dict.merge(meta_info.prefix("meta_info."))
            vis_dict = vis_dict.detach()
            return vis_dict

        # evaluate metrics
        metrics_all = self.evaluate_metrics(
            pred, targets, meta_info, self.metric_dict
        ).to_torch()
        out_dict = xdict()
        out_dict["imgname"] = meta_info["imgname"]
        out_dict.merge(ld_utils.prefix_dict(metrics_all, "metric."))

        if mode == "extract":
            mydict = xdict()
            mydict.merge(inputs.prefix("inputs."))
            mydict.merge(pred.prefix("pred."))
            mydict.merge(targets.prefix("targets."))
            mydict.merge(meta_info.prefix("meta_info."))
            mydict = mydict.detach()
            return mydict
        return out_dict, loss_dict

    def evaluate_metrics(self, pred, targets, meta_info, specs):
        metric_dict = xdict()
        for key in specs:
            metrics = eval_fn_dict[key](pred, targets, meta_info)
            metric_dict.merge(metrics)

        return metric_dict
