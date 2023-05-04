import os.path as op

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

# from src.datasets.tempo_dataset import TempoDataset
import common.ld_utils as ld_utils
import src.datasets.dataset_utils as dataset_utils
from src.datasets.arctic_dataset import ArcticDataset


def create_windows(imgnames, window_size):
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        my_chunks = [lst[i : i + n] for i in range(0, len(lst), n)]
        if len(my_chunks[-1]) == n:
            return my_chunks
        last_chunk = my_chunks[-1]
        last_element = last_chunk[-1]
        last_chunk_pad = [last_element for _ in range(n)]
        for idx, element in enumerate(last_chunk):
            last_chunk_pad[idx] = element
        my_chunks[-1] = last_chunk_pad
        return my_chunks

    img_seq_dict = {}
    for imgname in imgnames:
        sid, seq_name, view_idx, _ = imgname.split("/")[-4:]
        seq_name = "/".join([sid, seq_name, view_idx])
        if seq_name not in img_seq_dict.keys():
            img_seq_dict[seq_name] = []
        img_seq_dict[seq_name].append(imgname)

    windows = []
    for seq_name in img_seq_dict.keys():
        windows.append(chunks(sorted(img_seq_dict[seq_name]), window_size))

    windows = sum(windows, [])
    return windows


class TempoInferenceDataset(ArcticDataset):
    def _load_data(self, args, split):
        # load image features
        data_p = f"./data/arctic_data/data/feat/{args.img_feat_version}/{args.setup}_{split}.pt"
        assert op.exists(
            data_p
        ), f"Not found {data_p}; NOTE: only use ArcticDataset for single-frame model to evaluate and extract."
        logger.info(f"Loading {data_p}")
        data = torch.load(data_p)
        imgnames = data["imgnames"]
        vecs_list = data["feat_vec"]
        vec_dict = {}
        for imgname, vec in zip(imgnames, vecs_list):
            key = "/".join(imgname.split("/")[-4:])
            vec_dict[key] = vec
        self.vec_dict = vec_dict
        assert len(imgnames) == len(vec_dict.keys())

        # all imgnames for this split
        # override the original self.imgnames
        imgnames = [
            imgname.replace("/data/arctic_data/", "/arctic_data/")
            for imgname in imgnames
        ]
        self.imgnames = imgnames
        self.aug_data = False
        self.window_size = args.window_size

    def _process_imgnames(self, seq, split):
        imgnames = self.imgnames
        if seq is not None:
            imgnames = [imgname for imgname in imgnames if "/" + seq + "/" in imgname]
        assert len(imgnames) == len(set(imgnames))
        self.imgnames = imgnames

    def __init__(self, args, split, seq=None):
        Dataset.__init__(self)
        super()._load_data(args, split, seq)
        self._load_data(args, split)
        self._process_imgnames(seq, split)

        # split imgnames by windowsize into chunks
        # no overlappping frames btw chunks
        windows = create_windows(self.imgnames, self.window_size)
        windows = dataset_utils.downsample(windows, split)

        self.windows = windows
        num_imgnames = len(sum(self.windows, []))
        logger.info(
            f"TempoInferDataset Loaded {self.split} split, num samples {num_imgnames}"
        )

    def __getitem__(self, index):
        imgnames = self.windows[index]
        inputs_list = []
        targets_list = []
        meta_list = []
        img_feats = []
        load_rgb = not self.args.eval  # test.py do not load rgb
        for imgname in imgnames:
            short_imgname = "/".join(imgname.split("/")[-4:])
            # always load rgb because in training, we need to visualize
            # too complicated if not load rgb in eval or other situations
            # thus: load both rgb and features
            inputs, targets, meta_info = self.getitem(imgname, load_rgb=load_rgb)
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
        return inputs_list, targets_list, meta_list

    def __len__(self):
        return len(self.windows)
