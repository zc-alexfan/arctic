import os.path as op

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

import common.ld_utils as ld_utils
import src.datasets.dataset_utils as dataset_utils
from src.datasets.arctic_dataset import ArcticDataset


class TempoDataset(ArcticDataset):
    def _load_data(self, args, split):
        data_p = f"./data/arctic_data/data/feat/{args.img_feat_version}/{args.setup}_{split}.pt"
        logger.info(f"Loading: {data_p}")
        data = torch.load(data_p)
        imgnames = data["imgnames"]
        vecs_list = data["feat_vec"]
        assert len(imgnames) == len(vecs_list)
        vec_dict = {}
        for imgname, vec in zip(imgnames, vecs_list):
            key = "/".join(imgname.split("/")[-4:])
            vec_dict[key] = vec
        self.vec_dict = vec_dict

        assert len(imgnames) == len(vec_dict.keys())
        self.aug_data = False
        self.window_size = args.window_size

    def __init__(self, args, split, seq=None):
        Dataset.__init__(self)
        super()._load_data(args, split, seq)
        self._load_data(args, split)

        imgnames = list(self.vec_dict.keys())
        imgnames = dataset_utils.downsample(imgnames, split)

        self.imgnames = imgnames
        logger.info(
            f"TempoDataset Loaded {self.split} split, num samples {len(imgnames)}"
        )

    def __getitem__(self, index):
        imgname = self.imgnames[index]
        img_idx = int(op.basename(imgname).split(".")[0])
        ind = (
            np.arange(self.window_size) - (self.window_size - 1) / 2 + img_idx
        ).astype(np.int64)
        num_frames = self.data["/".join(imgname.split("/")[:2])]["params"][
            "rot_r"
        ].shape[0]
        ind = np.clip(
            ind, 10, num_frames - 10 - 1
        )  # skip first and last 10 frames as they are not useful
        imgnames = [op.join(op.dirname(imgname), "%05d.jpg" % (idx)) for idx in ind]

        targets_list = []
        meta_list = []
        img_feats = []
        inputs_list = []
        load_rgb = True if self.args.method in ["tempo_ft"] else False
        for imgname in imgnames:
            img_folder = f"./data/arctic_data/data/images/"
            inputs, targets, meta_info = self.getitem(
                op.join(img_folder, imgname), load_rgb=load_rgb
            )
            if load_rgb:
                inputs_list.append(inputs)
            else:
                img_feats.append(self.vec_dict[imgname].type(torch.FloatTensor))
            targets_list.append(targets)
            meta_list.append(meta_info)

        if load_rgb:
            inputs_list = ld_utils.stack_dl(
                ld_utils.ld2dl(inputs_list), dim=0, verbose=False
            )
            inputs = {"img": inputs_list["img"]}
        else:
            img_feats = torch.stack(img_feats, dim=0)
            inputs = {"img_feat": img_feats}

        targets_list = ld_utils.stack_dl(
            ld_utils.ld2dl(targets_list), dim=0, verbose=False
        )
        meta_list = ld_utils.stack_dl(ld_utils.ld2dl(meta_list), dim=0, verbose=False)

        targets_list["is_valid"] = torch.FloatTensor(np.array(targets_list["is_valid"]))
        targets_list["left_valid"] = torch.FloatTensor(
            np.array(targets_list["left_valid"])
        )
        targets_list["right_valid"] = torch.FloatTensor(
            np.array(targets_list["right_valid"])
        )
        targets_list["joints_valid_r"] = torch.FloatTensor(
            np.array(targets_list["joints_valid_r"])
        )
        targets_list["joints_valid_l"] = torch.FloatTensor(
            np.array(targets_list["joints_valid_l"])
        )
        meta_list["center"] = torch.FloatTensor(np.array(meta_list["center"]))
        meta_list["is_flipped"] = torch.FloatTensor(np.array(meta_list["is_flipped"]))
        meta_list["rot_angle"] = torch.FloatTensor(np.array(meta_list["rot_angle"]))
        return inputs, targets_list, meta_list
