import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim

import common.pl_utils as pl_utils
from common.comet_utils import log_dict
from common.pl_utils import avg_losses_cpu, push_checkpoint_metric
from common.xdict import xdict


class AbstractPL(pl.LightningModule):
    def __init__(
        self,
        args,
        push_images_fn,
        tracked_metric,
        metric_init_val,
        high_loss_val,
    ):
        super().__init__()
        self.experiment = args.experiment
        self.args = args
        self.tracked_metric = tracked_metric
        self.metric_init_val = metric_init_val

        self.started_training = False
        self.loss_dict_vec = []
        self.push_images = push_images_fn
        self.vis_train_batches = []
        self.vis_val_batches = []
        self.high_loss_val = high_loss_val
        self.max_vis_examples = 20
        self.val_step_outputs = []
        self.test_step_outputs = []

    def set_training_flags(self):
        self.started_training = True

    def load_from_ckpt(self, ckpt_path):
        sd = torch.load(ckpt_path)["state_dict"]
        print(self.load_state_dict(sd))

    def training_step(self, batch, batch_idx):
        self.set_training_flags()
        if len(self.vis_train_batches) < self.num_vis_train:
            self.vis_train_batches.append(batch)
        inputs, targets, meta_info = batch

        out = self.forward(inputs, targets, meta_info, "train")
        loss = out["loss"]

        loss = {k: loss[k].mean().view(-1) for k in loss}
        total_loss = sum(loss[k] for k in loss)

        loss_dict = {"total_loss": total_loss, "loss": total_loss}
        loss_dict.update(loss)

        for k, v in loss_dict.items():
            if k != "loss":
                loss_dict[k] = v.detach()

        log_every = self.args.log_every
        self.loss_dict_vec.append(loss_dict)
        self.loss_dict_vec = self.loss_dict_vec[len(self.loss_dict_vec) - log_every :]
        if batch_idx % log_every == 0 and batch_idx != 0:
            running_loss_dict = avg_losses_cpu(self.loss_dict_vec)
            running_loss_dict = xdict(running_loss_dict).postfix("__train")
            log_dict(self.experiment, running_loss_dict, step=self.global_step)
        return loss_dict

    def on_train_epoch_end(self):
        self.experiment.log_epoch_end(self.current_epoch)

    def validation_step(self, batch, batch_idx):
        if len(self.vis_val_batches) < self.num_vis_val:
            self.vis_val_batches.append(batch)
        out = self.inference_step(batch, batch_idx)
        self.val_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        outputs = self.val_step_outputs
        outputs = self.inference_epoch_end(outputs, postfix="__val")
        self.log("loss__val", outputs["loss__val"])
        self.val_step_outputs.clear()  # free memory
        return outputs

    def inference_step(self, batch, batch_idx):
        if self.training:
            self.eval()
        with torch.no_grad():
            inputs, targets, meta_info = batch
            out, loss = self.forward(inputs, targets, meta_info, "test")
            return {"out_dict": out, "loss": loss}

    def inference_epoch_end(self, out_list, postfix):
        if not self.started_training:
            self.started_training = True
            result = push_checkpoint_metric(self.tracked_metric, self.metric_init_val)
            return result

        # unpack
        outputs, loss_dict = pl_utils.reform_outputs(out_list)

        if "test" in postfix:
            per_img_metric_dict = {}
            for k, v in outputs.items():
                if "metric." in k:
                    per_img_metric_dict[k] = np.array(v)

        metric_dict = {}
        for k, v in outputs.items():
            if "metric." in k:
                metric_dict[k] = np.nanmean(np.array(v))

        loss_metric_dict = {}
        loss_metric_dict.update(metric_dict)
        loss_metric_dict.update(loss_dict)
        loss_metric_dict = xdict(loss_metric_dict).postfix(postfix)

        log_dict(
            self.experiment,
            loss_metric_dict,
            step=self.global_step,
        )

        if self.args.interface_p is None and "test" not in postfix:
            result = push_checkpoint_metric(
                self.tracked_metric, loss_metric_dict[self.tracked_metric]
            )
            self.log(self.tracked_metric, result[self.tracked_metric])

        if not self.args.no_vis:
            print("Rendering train images")
            self.visualize_batches(self.vis_train_batches, "_train", False)
            print("Rendering val images")
            self.visualize_batches(self.vis_val_batches, "_val", False)

        if "test" in postfix:
            return (
                outputs,
                {"per_img_metric_dict": per_img_metric_dict},
                metric_dict,
            )
        return loss_metric_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, self.args.lr_dec_epoch, gamma=self.args.lr_decay, verbose=True
        )
        return [optimizer], [scheduler]

    def visualize_batches(self, batches, postfix, no_tqdm=True):
        im_list = []
        if self.training:
            self.eval()

        tic = time.time()
        for batch_idx, batch in enumerate(batches):
            with torch.no_grad():
                inputs, targets, meta_info = batch
                vis_dict = self.forward(inputs, targets, meta_info, "vis")
                for vis_fn in self.vis_fns:
                    curr_im_list = vis_fn(
                        vis_dict,
                        self.max_vis_examples,
                        self.renderer,
                        postfix=postfix,
                        no_tqdm=no_tqdm,
                    )
                    im_list += curr_im_list
                print("Rendering: %d/%d" % (batch_idx + 1, len(batches)))

        self.push_images(self.experiment, im_list, self.global_step)
        print("Done rendering (%.1fs)" % (time.time() - tic))
        return im_list
