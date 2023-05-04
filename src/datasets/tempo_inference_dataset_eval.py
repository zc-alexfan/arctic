import numpy as np
import torch

import common.ld_utils as ld_utils
from src.datasets.tempo_inference_dataset import TempoInferenceDataset


class TempoInferenceDatasetEval(TempoInferenceDataset):
    def __getitem__(self, index):
        imgnames = self.windows[index]
        inputs_list = []
        targets_list = []
        meta_list = []
        img_feats = []
        # load_rgb = not self.args.eval  # test.py do not load rgb
        load_rgb = False
        for imgname in imgnames:
            short_imgname = "/".join(imgname.split("/")[-4:])
            # always load rgb because in training, we need to visualize
            # too complicated if not load rgb in eval or other situations
            # thus: load both rgb and features
            inputs, targets, meta_info = self.getitem_eval(imgname, load_rgb=load_rgb)
            img_feats.append(self.vec_dict[short_imgname])
            inputs_list.append(inputs)
            targets_list.append(targets)
            meta_list.append(meta_info)

        if load_rgb:
            inputs_list = ld_utils.stack_dl(
                ld_utils.ld2dl(inputs_list), dim=0, verbose=False
            )
        else:
            inputs_list = {}
        targets_list = ld_utils.stack_dl(
            ld_utils.ld2dl(targets_list), dim=0, verbose=False
        )
        meta_list = ld_utils.stack_dl(ld_utils.ld2dl(meta_list), dim=0, verbose=False)
        img_feats = torch.stack(img_feats, dim=0).float()

        inputs_list["img_feat"] = img_feats
        meta_list["center"] = torch.FloatTensor(np.array(meta_list["center"]))
        meta_list["is_flipped"] = torch.FloatTensor(np.array(meta_list["is_flipped"]))
        meta_list["rot_angle"] = torch.FloatTensor(np.array(meta_list["rot_angle"]))
        return inputs_list, targets_list, meta_list
