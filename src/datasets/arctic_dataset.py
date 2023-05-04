import json
import os.path as op

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

import common.data_utils as data_utils
import common.rot as rot
import common.transforms as tf
import src.datasets.dataset_utils as dataset_utils
from common.data_utils import read_img
from common.object_tensors import ObjectTensors
from src.datasets.dataset_utils import get_valid, pad_jts2d


class ArcticDataset(Dataset):
    def __getitem__(self, index):
        imgname = self.imgnames[index]
        data = self.getitem(imgname)
        return data

    def getitem(self, imgname, load_rgb=True):
        args = self.args
        # LOADING START
        speedup = args.speedup
        sid, seq_name, view_idx, image_idx = imgname.split("/")[-4:]
        obj_name = seq_name.split("_")[0]
        view_idx = int(view_idx)

        seq_data = self.data[f"{sid}/{seq_name}"]

        data_cam = seq_data["cam_coord"]
        data_2d = seq_data["2d"]
        data_bbox = seq_data["bbox"]
        data_params = seq_data["params"]

        vidx = int(image_idx.split(".")[0]) - self.ioi_offset[sid]
        vidx, is_valid, right_valid, left_valid = get_valid(
            data_2d, data_cam, vidx, view_idx, imgname
        )

        if view_idx == 0:
            intrx = data_params["K_ego"][vidx].copy()
        else:
            intrx = np.array(self.intris_mat[sid][view_idx - 1])

        # hands
        joints2d_r = pad_jts2d(data_2d["joints.right"][vidx, view_idx].copy())
        joints3d_r = data_cam["joints.right"][vidx, view_idx].copy()

        joints2d_l = pad_jts2d(data_2d["joints.left"][vidx, view_idx].copy())
        joints3d_l = data_cam["joints.left"][vidx, view_idx].copy()

        pose_r = data_params["pose_r"][vidx].copy()
        betas_r = data_params["shape_r"][vidx].copy()
        pose_l = data_params["pose_l"][vidx].copy()
        betas_l = data_params["shape_l"][vidx].copy()

        # distortion parameters for egocam rendering
        dist = data_params["dist"][vidx].copy()
        # NOTE:
        # kp2d, kp3d are in undistored space
        # thus, results for evaluation is in the undistorted space (non-curved)
        # dist parameters can be used for rendering in visualization

        # objects
        bbox2d = pad_jts2d(data_2d["bbox3d"][vidx, view_idx].copy())
        bbox3d = data_cam["bbox3d"][vidx, view_idx].copy()
        bbox2d_t = bbox2d[:8]
        bbox2d_b = bbox2d[8:]
        bbox3d_t = bbox3d[:8]
        bbox3d_b = bbox3d[8:]

        kp2d = pad_jts2d(data_2d["kp3d"][vidx, view_idx].copy())
        kp3d = data_cam["kp3d"][vidx, view_idx].copy()
        kp2d_t = kp2d[:16]
        kp2d_b = kp2d[16:]
        kp3d_t = kp3d[:16]
        kp3d_b = kp3d[16:]

        obj_radian = data_params["obj_arti"][vidx].copy()

        image_size = self.image_sizes[sid][view_idx]
        image_size = {"width": image_size[0], "height": image_size[1]}

        bbox = data_bbox[vidx, view_idx]  # original bbox
        is_egocam = "/0/" in imgname

        # LOADING END

        # SPEEDUP PROCESS
        (
            joints2d_r,
            joints2d_l,
            kp2d_b,
            kp2d_t,
            bbox2d_b,
            bbox2d_t,
            bbox,
        ) = dataset_utils.transform_2d_for_speedup(
            speedup,
            is_egocam,
            joints2d_r,
            joints2d_l,
            kp2d_b,
            kp2d_t,
            bbox2d_b,
            bbox2d_t,
            bbox,
            args.ego_image_scale,
        )
        img_status = True
        if load_rgb:
            if speedup:
                imgname = imgname.replace("/images/", "/cropped_images/")
            imgname = imgname.replace(
                "/arctic_data/", "/data/arctic_data/data/"
            ).replace("/data/data/", "/data/")
            # imgname = imgname.replace("/arctic_data/", "/data/arctic_data/")
            cv_img, img_status = read_img(imgname, (2800, 2000, 3))
        else:
            norm_img = None

        center = [bbox[0], bbox[1]]
        scale = bbox[2]

        # augment parameters
        augm_dict = data_utils.augm_params(
            self.aug_data,
            args.flip_prob,
            args.noise_factor,
            args.rot_factor,
            args.scale_factor,
        )

        use_gt_k = args.use_gt_k
        if is_egocam:
            # no scaling for egocam to make intrinsics consistent
            use_gt_k = True
            augm_dict["sc"] = 1.0

        joints2d_r = data_utils.j2d_processing(
            joints2d_r, center, scale, augm_dict, args.img_res
        )
        joints2d_l = data_utils.j2d_processing(
            joints2d_l, center, scale, augm_dict, args.img_res
        )
        kp2d_b = data_utils.j2d_processing(
            kp2d_b, center, scale, augm_dict, args.img_res
        )
        kp2d_t = data_utils.j2d_processing(
            kp2d_t, center, scale, augm_dict, args.img_res
        )
        bbox2d_b = data_utils.j2d_processing(
            bbox2d_b, center, scale, augm_dict, args.img_res
        )
        bbox2d_t = data_utils.j2d_processing(
            bbox2d_t, center, scale, augm_dict, args.img_res
        )
        bbox2d = np.concatenate((bbox2d_t, bbox2d_b), axis=0)
        kp2d = np.concatenate((kp2d_t, kp2d_b), axis=0)

        # data augmentation: image
        if load_rgb:
            img = data_utils.rgb_processing(
                self.aug_data,
                cv_img,
                center,
                scale,
                augm_dict,
                img_res=args.img_res,
            )
            img = torch.from_numpy(img).float()
            norm_img = self.normalize_img(img)

        # exporting starts
        inputs = {}
        targets = {}
        meta_info = {}
        inputs["img"] = norm_img
        meta_info["imgname"] = imgname
        rot_r = data_cam["rot_r_cam"][vidx, view_idx]
        rot_l = data_cam["rot_l_cam"][vidx, view_idx]

        pose_r = np.concatenate((rot_r, pose_r), axis=0)
        pose_l = np.concatenate((rot_l, pose_l), axis=0)

        # hands
        targets["mano.pose.r"] = torch.from_numpy(
            data_utils.pose_processing(pose_r, augm_dict)
        ).float()
        targets["mano.pose.l"] = torch.from_numpy(
            data_utils.pose_processing(pose_l, augm_dict)
        ).float()
        targets["mano.beta.r"] = torch.from_numpy(betas_r).float()
        targets["mano.beta.l"] = torch.from_numpy(betas_l).float()
        targets["mano.j2d.norm.r"] = torch.from_numpy(joints2d_r[:, :2]).float()
        targets["mano.j2d.norm.l"] = torch.from_numpy(joints2d_l[:, :2]).float()

        # object
        targets["object.kp3d.full.b"] = torch.from_numpy(kp3d_b[:, :3]).float()
        targets["object.kp2d.norm.b"] = torch.from_numpy(kp2d_b[:, :2]).float()
        targets["object.kp3d.full.t"] = torch.from_numpy(kp3d_t[:, :3]).float()
        targets["object.kp2d.norm.t"] = torch.from_numpy(kp2d_t[:, :2]).float()

        targets["object.bbox3d.full.b"] = torch.from_numpy(bbox3d_b[:, :3]).float()
        targets["object.bbox2d.norm.b"] = torch.from_numpy(bbox2d_b[:, :2]).float()
        targets["object.bbox3d.full.t"] = torch.from_numpy(bbox3d_t[:, :3]).float()
        targets["object.bbox2d.norm.t"] = torch.from_numpy(bbox2d_t[:, :2]).float()
        targets["object.radian"] = torch.FloatTensor(np.array(obj_radian))

        targets["object.kp2d.norm"] = torch.from_numpy(kp2d[:, :2]).float()
        targets["object.bbox2d.norm"] = torch.from_numpy(bbox2d[:, :2]).float()

        # compute RT from cano space to augmented space
        # this transform match j3d processing
        obj_idx = self.obj_names.index(obj_name)
        meta_info["kp3d.cano"] = self.kp3d_cano[obj_idx] / 1000  # meter
        kp3d_cano = meta_info["kp3d.cano"].numpy()
        kp3d_target = targets["object.kp3d.full.b"][:, :3].numpy()

        # rotate canonical kp3d to match original image
        R, _ = tf.solve_rigid_tf_np(kp3d_cano, kp3d_target)
        obj_rot = (
            rot.batch_rot2aa(torch.from_numpy(R).float().view(1, 3, 3)).view(3).numpy()
        )

        # multiply rotation from data augmentation
        obj_rot_aug = rot.rot_aa(obj_rot, augm_dict["rot"])
        targets["object.rot"] = torch.FloatTensor(obj_rot_aug).view(1, 3)

        # full image camera coord
        targets["mano.j3d.full.r"] = torch.FloatTensor(joints3d_r[:, :3])
        targets["mano.j3d.full.l"] = torch.FloatTensor(joints3d_l[:, :3])
        targets["object.kp3d.full.b"] = torch.FloatTensor(kp3d_b[:, :3])

        meta_info["query_names"] = obj_name
        meta_info["window_size"] = torch.LongTensor(np.array([args.window_size]))

        # scale and center in the original image space
        scale_original = max([image_size["width"], image_size["height"]]) / 200.0
        center_original = [image_size["width"] / 2.0, image_size["height"] / 2.0]
        intrx = data_utils.get_aug_intrix(
            intrx,
            args.focal_length,
            args.img_res,
            use_gt_k,
            center_original[0],
            center_original[1],
            augm_dict["sc"] * scale_original,
        )

        if is_egocam and self.egocam_k is None:
            self.egocam_k = intrx
        elif is_egocam and self.egocam_k is not None:
            intrx = self.egocam_k

        meta_info["intrinsics"] = torch.FloatTensor(intrx)
        if not is_egocam:
            dist = dist * float("nan")
        meta_info["dist"] = torch.FloatTensor(dist)
        meta_info["center"] = np.array(center, dtype=np.float32)
        meta_info["is_flipped"] = augm_dict["flip"]
        meta_info["rot_angle"] = np.float32(augm_dict["rot"])
        # meta_info["sample_index"] = index

        # root and at least 3 joints inside image
        targets["is_valid"] = float(is_valid)
        targets["left_valid"] = float(left_valid) * float(is_valid)
        targets["right_valid"] = float(right_valid) * float(is_valid)
        targets["joints_valid_r"] = np.ones(21) * targets["right_valid"]
        targets["joints_valid_l"] = np.ones(21) * targets["left_valid"]

        return inputs, targets, meta_info

    def _process_imgnames(self, seq, split):
        imgnames = self.imgnames
        if seq is not None:
            imgnames = [imgname for imgname in imgnames if "/" + seq + "/" in imgname]
        assert len(imgnames) == len(set(imgnames))
        imgnames = dataset_utils.downsample(imgnames, split)
        self.imgnames = imgnames

    def _load_data(self, args, split, seq):
        self.args = args
        self.split = split
        self.aug_data = split.endswith("train")
        # during inference, turn off
        if seq is not None:
            self.aug_data = False
        self.normalize_img = Normalize(mean=args.img_norm_mean, std=args.img_norm_std)

        if "train" in split:
            self.mode = "train"
        elif "val" in split:
            self.mode = "val"
        elif "test" in split:
            self.mode = "test"

        short_split = split.replace("mini", "").replace("tiny", "").replace("small", "")
        data_p = op.join(
            f"./data/arctic_data/data/splits/{args.setup}_{short_split}.npy"
        )
        logger.info(f"Loading {data_p}")
        data = np.load(data_p, allow_pickle=True).item()

        self.data = data["data_dict"]
        self.imgnames = data["imgnames"]

        with open("./data/arctic_data/data/meta/misc.json", "r") as f:
            misc = json.load(f)

        # unpack
        subjects = list(misc.keys())
        intris_mat = {}
        world2cam = {}
        image_sizes = {}
        ioi_offset = {}
        for subject in subjects:
            world2cam[subject] = misc[subject]["world2cam"]
            intris_mat[subject] = misc[subject]["intris_mat"]
            image_sizes[subject] = misc[subject]["image_size"]
            ioi_offset[subject] = misc[subject]["ioi_offset"]

        self.world2cam = world2cam
        self.intris_mat = intris_mat
        self.image_sizes = image_sizes
        self.ioi_offset = ioi_offset

        object_tensors = ObjectTensors()
        self.kp3d_cano = object_tensors.obj_tensors["kp_bottom"]
        self.obj_names = object_tensors.obj_tensors["names"]
        self.egocam_k = None

    def __init__(self, args, split, seq=None):
        self._load_data(args, split, seq)
        self._process_imgnames(seq, split)
        logger.info(
            f"ImageDataset Loaded {self.split} split, num samples {len(self.imgnames)}"
        )

    def __len__(self):
        return len(self.imgnames)

    def getitem_eval(self, imgname, load_rgb=True):
        args = self.args
        # LOADING START
        speedup = args.speedup
        sid, seq_name, view_idx, image_idx = imgname.split("/")[-4:]
        obj_name = seq_name.split("_")[0]
        view_idx = int(view_idx)

        seq_data = self.data[f"{sid}/{seq_name}"]

        data_bbox = seq_data["bbox"]
        data_params = seq_data["params"]

        vidx = int(image_idx.split(".")[0]) - self.ioi_offset[sid]

        if view_idx == 0:
            intrx = data_params["K_ego"][vidx].copy()
        else:
            intrx = np.array(self.intris_mat[sid][view_idx - 1])

        # distortion parameters for egocam rendering
        dist = data_params["dist"][vidx].copy()

        bbox = data_bbox[vidx, view_idx]  # original bbox
        is_egocam = "/0/" in imgname

        image_size = self.image_sizes[sid][view_idx]
        image_size = {"width": image_size[0], "height": image_size[1]}

        # SPEEDUP PROCESS
        bbox = dataset_utils.transform_bbox_for_speedup(
            speedup,
            is_egocam,
            bbox,
            args.ego_image_scale,
        )
        img_status = True
        if load_rgb:
            if speedup:
                imgname = imgname.replace("/images/", "/cropped_images/")
            imgname = imgname.replace("/arctic_data/", "/data/arctic_data/data/")
            cv_img, img_status = read_img(imgname, (2800, 2000, 3))
        else:
            norm_img = None

        center = [bbox[0], bbox[1]]
        scale = bbox[2]
        self.aug_data = False

        # augment parameters
        augm_dict = data_utils.augm_params(
            self.aug_data,
            args.flip_prob,
            args.noise_factor,
            args.rot_factor,
            args.scale_factor,
        )

        use_gt_k = args.use_gt_k
        if is_egocam:
            # no scaling for egocam to make intrinsics consistent
            use_gt_k = True
            augm_dict["sc"] = 1.0

        # data augmentation: image
        if load_rgb:
            img = data_utils.rgb_processing(
                self.aug_data,
                cv_img,
                center,
                scale,
                augm_dict,
                img_res=args.img_res,
            )
            img = torch.from_numpy(img).float()
            norm_img = self.normalize_img(img)

        # exporting starts
        inputs = {}
        targets = {}
        meta_info = {}
        inputs["img"] = norm_img
        meta_info["imgname"] = imgname

        meta_info["query_names"] = obj_name
        meta_info["window_size"] = torch.LongTensor(np.array([args.window_size]))

        # scale and center in the original image space
        scale_original = max([image_size["width"], image_size["height"]]) / 200.0
        center_original = [image_size["width"] / 2.0, image_size["height"] / 2.0]
        intrx = data_utils.get_aug_intrix(
            intrx,
            args.focal_length,
            args.img_res,
            use_gt_k,
            center_original[0],
            center_original[1],
            augm_dict["sc"] * scale_original,
        )

        if is_egocam and self.egocam_k is None:
            self.egocam_k = intrx
        elif is_egocam and self.egocam_k is not None:
            intrx = self.egocam_k

        meta_info["intrinsics"] = torch.FloatTensor(intrx)
        if not is_egocam:
            dist = dist * float("nan")

        meta_info["dist"] = torch.FloatTensor(dist)
        meta_info["center"] = np.array(center, dtype=np.float32)
        meta_info["is_flipped"] = augm_dict["flip"]
        meta_info["rot_angle"] = np.float32(augm_dict["rot"])
        return inputs, targets, meta_info
