import os
import os.path as op

import numpy as np
import torch
from loguru import logger
from PIL import Image
from pytorch3d.transforms import matrix_to_axis_angle
from torch.utils.data import DataLoader
from tqdm import tqdm

import common.ld_utils as ld_utils
import common.thing as thing
from common.data_utils import denormalize_images
from common.xdict import xdict
from src.callbacks.process.process_generic import prepare_interfield
from src.datasets.arctic_dataset import ArcticDataset


def prepare_data(full_seq_name, exp_key, data_keys, layers, device, task, eval_p):
    sid = full_seq_name[:3]
    view = full_seq_name[-1]
    folder_p = op.join(eval_p, full_seq_name)
    if "pose" in task and "submit_" in task:
        gt_folder = op.join("./eval_server_gt_pose", full_seq_name)
    elif "field" in task and "submit_" in task:
        gt_folder = op.join("./eval_server_gt_field", full_seq_name)
    else:
        gt_folder = folder_p
    logger.info(f"Reading keys: {data_keys}")
    batch = read_keys(gt_folder, folder_p, keys=data_keys, verbose=False)
    batch = xdict(batch)
    logger.info("Done")

    # trim the frames at the end
    # as in pred, they were a multiple of `window_size
    num_gts = len(batch["meta_info.imgname"])
    for key in data_keys:
        if "pred." in key or "targets." in key:
            batch.overwrite(key, batch[key][:num_gts])

    # assert right seq
    imgnames = batch["meta_info.imgname"]
    for imgname in imgnames:
        curr_sid, curr_seq, curr_view, _ = imgname.split("/")[-4:]
        assert full_seq_name[4:-2] == curr_seq
        assert curr_sid == sid
        assert curr_view == view

    if "pose" in task:
        batch.overwrite(
            "pred.mano.pose.r", matrix_to_axis_angle(batch["pred.mano.pose.r"])
        )
        batch.overwrite(
            "pred.mano.pose.l", matrix_to_axis_angle(batch["pred.mano.pose.l"])
        )

        logger.info("forward params")
        batch = fk_params_batch(batch, layers, device, flag="pred")
        batch = fk_params_batch(batch, layers, device, flag="targets")
        logger.info("Done")

        meta_info = batch.search("meta_info.", replace_to="")
        pred = batch.search("pred.", replace_to="")
        targets = batch.search("targets.", replace_to="")
        meta_info.overwrite("part_ids", targets["object.parts_ids"])
        meta_info.overwrite("diameter", targets["object.diameter"])

        logger.info("prepare interfield")
        targets = prepare_interfield(targets, max_dist=0.1)
        logger.info("Done")

    elif "field" in task:
        logger.info("forward params")
        batch = fk_params_batch(batch, layers, device, flag="targets")
        logger.info("Done")

        meta_info = batch.search("meta_info.", replace_to="")
        pred = batch.search("pred.", replace_to="")
        targets = batch.search("targets.", replace_to="")
        meta_info["object.v_len"] = targets["object.v_len"]

        logger.info("prepare interfield")
        targets = prepare_interfield(targets, max_dist=0.1)
        logger.info("Done")

    data = xdict()
    data.merge(pred.prefix("pred."))
    data.merge(targets.prefix("targets."))
    data.merge(meta_info.prefix("meta_info."))
    data = data.to("cpu")
    return data


def fk_params_batch(batch, layers, device, flag):
    mano_r = layers["right"]
    mano_l = layers["left"]
    object_tensors = layers["object_tensors"]
    batch = xdict(thing.thing2dev(dict(batch), device))
    pose_r = batch[f"{flag}.mano.pose.r"].reshape(-1, 48)
    pose_l = batch[f"{flag}.mano.pose.l"].reshape(-1, 48)
    cam_r = batch[f"{flag}.mano.cam_t.r"].view(-1, 1, 3)
    cam_l = batch[f"{flag}.mano.cam_t.l"].view(-1, 1, 3)
    cam_o = batch[f"{flag}.object.cam_t"].view(-1, 1, 3)

    out_r = mano_r(
        global_orient=pose_r[:, :3].reshape(-1, 3),
        hand_pose=pose_r[:, 3:].reshape(-1, 45),
        betas=batch[f"{flag}.mano.beta.r"].view(-1, 10),
    )

    out_l = mano_l(
        global_orient=pose_l[:, :3].reshape(-1, 3),
        hand_pose=pose_l[:, 3:].reshape(-1, 45),
        betas=batch[f"{flag}.mano.beta.l"].view(-1, 10),
    )
    query_names = batch["meta_info.query_names"]
    out_o = object_tensors.forward(
        batch[f"{flag}.object.radian"].view(-1, 1),
        batch[f"{flag}.object.rot"].view(-1, 3),
        None,
        query_names,
    )
    v3d_r = out_r.vertices + cam_r
    v3d_l = out_l.vertices + cam_l
    v3d_o = out_o["v"] + cam_o
    j3d_r = out_r.joints + cam_r
    j3d_l = out_l.joints + cam_l
    out = {
        f"{flag}.mano.v3d.cam.r": v3d_r,
        f"{flag}.mano.v3d.cam.l": v3d_l,
        f"{flag}.mano.j3d.cam.r": j3d_r,
        f"{flag}.mano.j3d.cam.l": j3d_l,
        f"{flag}.object.v.cam": v3d_o,
        f"{flag}.object.v_len": out_o["v_len"],
        f"{flag}.object.diameter": out_o["diameter"],
        f"{flag}.object.parts_ids": out_o["parts_ids"],
    }
    batch.merge(out)
    return batch


def read_keys(gt_folder_p, folder_p, keys, verbose=True):
    out = {}

    if verbose:
        pbar = tqdm(keys)
    else:
        pbar = keys
    for key in pbar:
        if verbose:
            pbar.set_description(f"Loading {key}")
        if key in "inputs.img":
            # skip images
            continue
        if "targets." in key or "meta_info." in key or "inputs." in key:
            curr_folder_p = op.join(gt_folder_p, key.split(".")[0])
        else:
            curr_folder_p = op.join(folder_p, "preds")
        data_p = op.join(curr_folder_p, key + ".pt")
        # print(data_p)
        data = torch.load(data_p)
        if isinstance(data, (torch.HalfTensor, torch.cuda.HalfTensor)):
            data = data.type(torch.float32)
        out[key] = data
    return out


def save_results(out, out_dir):
    # interface.verify_interface(out)
    for seq_name, seq_data in out.items():
        out_folder = op.join(out_dir, seq_name)
        exp_key, seq_name = out_folder.split("/")[-2:]

        input_p = op.join(out_folder, "inputs")
        target_p = op.join(out_folder, "targets")
        meta_p = op.join(out_folder, "meta_info")
        pred_p = op.join(out_folder, "preds")
        img_p = op.join(out_folder, "images")

        logger.info(f"Dumping pose est results at {out_folder}")
        for key, val in seq_data.items():
            if "inputs.img" in key:
                # save images
                imgs = denormalize_images(val)
                images = (imgs.permute(0, 2, 3, 1) * 255).numpy().astype(np.uint8)
                for idx, img in tqdm(enumerate(images), total=len(images)):
                    im = Image.fromarray(img)
                    out_p = op.join(img_p, "%05d.jpg" % (idx))
                    os.makedirs(op.dirname(out_p), exist_ok=True)
                    im.save(out_p)
            else:
                if "inputs." in key:
                    out_p = op.join(input_p, key + ".pt")
                elif "targets." in key:
                    out_p = op.join(target_p, key + ".pt")
                elif "meta_info." in key:
                    out_p = op.join(meta_p, key + ".pt")
                elif "pred." in key:
                    out_p = op.join(pred_p, key + ".pt")
                else:
                    print(f"Skipping {key} of type {type(val)}")

                os.makedirs(op.dirname(out_p), exist_ok=True)
                if isinstance(
                    val, (torch.FloatTensor, torch.cuda.FloatTensor)
                ) and key not in ["pred.feat_vec"]:
                    # reduce storage requirement
                    val = val.type(torch.float16)
                print(f"Saving {key} to {out_p}")
                torch.save(val, out_p)


def std_interface(out_list):
    out_list = ld_utils.ld2dl(out_list)
    out = ld_utils.cat_dl(out_list, 0)
    for key, val in out.items():
        if isinstance(val, torch.Tensor):
            out[key] = val.squeeze()

    # verify that all keys have same length
    keys = list(out.keys())
    key0 = keys[0]
    for key in keys:
        if len(out[key]) != len(out[key0]):
            logger.info(
                f"Key {key} has length {len(out[key])} while {key0} has length {len(out[key0])}"
            )
            import ipdb

            ipdb.set_trace()

    # sort by imgname
    imgnames = np.array(out["meta_info.imgname"])
    num_examples = len(set(out["meta_info.imgname"]))
    sort_idx = np.argsort(imgnames)
    for key, val in out.items():
        assert len(val) == len(sort_idx)
        if isinstance(val, (torch.Tensor, np.ndarray)):
            out[key] = val[sort_idx]
        elif isinstance(val, (list)):
            out[key] = [val[idx] for idx in sort_idx]
        else:
            print(f"Skipping {key} of type {type(out)}")

    # split according to camera
    imgnames = np.array(out["meta_info.imgname"])
    cam_ids = []
    all_seqs = []
    for imgname in imgnames:
        sid, seq_name, cam_id, frame = imgname.split("/")[-4:]
        all_seqs.append(seq_name)
        cam_ids.append(int(cam_id))

    assert len(set(all_seqs)) == 1
    cam_ids = np.array(cam_ids)
    all_cams = list(set(cam_ids))
    out_cam = {}
    imgnames_one = imgnames.reshape(len(all_cams), -1)
    num_examples = len(set(imgnames_one[0].tolist()))
    for cam_id in all_cams:
        sub_idx = np.where(cam_id == cam_ids)[0][:num_examples]
        curr_cam_out = {}
        for key, val in out.items():
            if isinstance(val, (torch.Tensor, np.ndarray)):
                curr_cam_out[key] = val[sub_idx]
            elif isinstance(val, (list)):
                curr_cam_out[key] = [val[idx] for idx in sub_idx]
            else:
                print(f"Skipping {key} of type {type(out)}")

            assert len(curr_cam_out[key]) == num_examples
        out_cam[f"{sid}_{seq_name}_{cam_id}"] = curr_cam_out
    return out_cam


def fetch_dataset(args, seq):
    ds = ArcticDataset(args=args, split=args.run_on, seq=seq)
    return ds


def fetch_dataloader(args, seq):
    dataset = fetch_dataset(args, seq)
    return DataLoader(
        dataset=dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
