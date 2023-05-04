import numpy as np
import torch


class PreprocessDataset(torch.utils.data.Dataset):
    def __init__(self, bundle):
        self.rot_r = bundle["rot_r"]
        self.pose_r = bundle["pose_r"]
        self.trans_r = bundle["trans_r"]
        self.shape_r = bundle["shape_r"]
        self.fitting_err_r = bundle["fitting_err_r"]

        self.rot_l = bundle["rot_l"]
        self.pose_l = bundle["pose_l"]
        self.trans_l = bundle["trans_l"]
        self.shape_l = bundle["shape_l"]
        self.fitting_err_l = bundle["fitting_err_l"]

        # smplx
        self.smplx_transl = bundle["smplx_transl"]
        self.smplx_global_orient = bundle["smplx_global_orient"]
        self.smplx_body_pose = bundle["smplx_body_pose"]
        self.smplx_jaw_pose = bundle["smplx_jaw_pose"]
        self.smplx_leye_pose = bundle["smplx_leye_pose"]
        self.smplx_reye_pose = bundle["smplx_reye_pose"]
        self.smplx_left_hand_pose = bundle["smplx_left_hand_pose"]
        self.smplx_right_hand_pose = bundle["smplx_right_hand_pose"]

        self.obj_arti = bundle["obj_params"][:, 0]  # radian
        self.obj_rot = bundle["obj_params"][:, 1:4]
        self.obj_trans = bundle["obj_params"][:, 4:]

        self.world2ego = bundle["world2ego"]
        self.K_ego = bundle["K_ego"]
        self.dist = bundle["dist"]
        self.obj_name = bundle["obj_name"]

    def __getitem__(self, idx):
        out_dict = {}

        out_dict["rot_r"] = self.rot_r[idx]
        out_dict["pose_r"] = self.pose_r[idx]
        out_dict["trans_r"] = self.trans_r[idx]
        out_dict["shape_r"] = self.shape_r[idx]
        out_dict["fitting_err_r"] = self.fitting_err_r[idx]

        out_dict["rot_l"] = self.rot_l[idx]
        out_dict["pose_l"] = self.pose_l[idx]
        out_dict["trans_l"] = self.trans_l[idx]
        out_dict["shape_l"] = self.shape_l[idx]
        out_dict["fitting_err_l"] = self.fitting_err_l[idx]

        # smplx
        out_dict["smplx_transl"] = self.smplx_transl[idx]
        out_dict["smplx_global_orient"] = self.smplx_global_orient[idx]
        out_dict["smplx_body_pose"] = self.smplx_body_pose[idx]
        out_dict["smplx_jaw_pose"] = self.smplx_jaw_pose[idx]
        out_dict["smplx_leye_pose"] = self.smplx_leye_pose[idx]
        out_dict["smplx_reye_pose"] = self.smplx_reye_pose[idx]
        out_dict["smplx_left_hand_pose"] = self.smplx_left_hand_pose[idx]
        out_dict["smplx_right_hand_pose"] = self.smplx_right_hand_pose[idx]

        out_dict["obj_arti"] = self.obj_arti[idx]
        out_dict["obj_rot"] = self.obj_rot[idx]
        out_dict["obj_trans"] = self.obj_trans[idx]  # to meter

        out_dict["world2ego"] = self.world2ego[idx]
        out_dict["dist"] = self.dist
        out_dict["K_ego"] = self.K_ego
        out_dict["query_names"] = self.obj_name
        return out_dict

    def __len__(self):
        return self.rot_r.shape[0]


def construct_loader(mano_p):
    obj_p = mano_p.replace(".mano.", ".object.")
    ego_p = mano_p.replace(".mano.", ".egocam.dist.")

    # MANO
    data = np.load(
        mano_p,
        allow_pickle=True,
    ).item()

    num_frames = len(data["right"]["rot"])

    rot_r = torch.FloatTensor(data["right"]["rot"])
    pose_r = torch.FloatTensor(data["right"]["pose"])
    trans_r = torch.FloatTensor(data["right"]["trans"])
    shape_r = torch.FloatTensor(data["right"]["shape"]).repeat(num_frames, 1)
    fitting_err_r = data["right"]["fitting_err"]

    rot_l = torch.FloatTensor(data["left"]["rot"])
    pose_l = torch.FloatTensor(data["left"]["pose"])
    trans_l = torch.FloatTensor(data["left"]["trans"])
    shape_l = torch.FloatTensor(data["left"]["shape"]).repeat(num_frames, 1)
    fitting_err_l = data["left"]["fitting_err"]
    assert len(fitting_err_l) > 50, f"Failed: {mano_p}"
    assert len(fitting_err_r) > 50, f"Failed: {mano_p}"

    obj_params = torch.FloatTensor(np.load(obj_p, allow_pickle=True))
    assert rot_r.shape[0] == obj_params.shape[0]

    obj_name = obj_p.split("/")[-1].split("_")[0]

    ego_p = mano_p.replace("mano.npy", "egocam.dist.npy")
    egocam = np.load(ego_p, allow_pickle=True).item()
    R_ego = torch.FloatTensor(egocam["R_k_cam_np"])
    T_ego = torch.FloatTensor(egocam["T_k_cam_np"])
    K_ego = torch.FloatTensor(egocam["intrinsics"])
    dist = torch.FloatTensor(egocam["dist8"])

    num_frames = R_ego.shape[0]
    world2ego = torch.zeros((num_frames, 4, 4))
    world2ego[:, :3, :3] = R_ego
    world2ego[:, :3, 3] = T_ego.view(num_frames, 3)
    world2ego[:, 3, 3] = 1

    assert torch.isnan(obj_params).sum() == 0
    assert torch.isinf(obj_params).sum() == 0

    # smplx
    smplx_p = mano_p.replace(".mano.", ".smplx.")
    smplx_data = np.load(smplx_p, allow_pickle=True).item()

    smplx_transl = torch.FloatTensor(smplx_data["transl"])
    smplx_global_orient = torch.FloatTensor(smplx_data["global_orient"])
    smplx_body_pose = torch.FloatTensor(smplx_data["body_pose"])
    smplx_jaw_pose = torch.FloatTensor(smplx_data["jaw_pose"])
    smplx_leye_pose = torch.FloatTensor(smplx_data["leye_pose"])
    smplx_reye_pose = torch.FloatTensor(smplx_data["reye_pose"])
    smplx_left_hand_pose = torch.FloatTensor(smplx_data["left_hand_pose"])
    smplx_right_hand_pose = torch.FloatTensor(smplx_data["right_hand_pose"])

    bundle = {}
    bundle["rot_r"] = rot_r
    bundle["pose_r"] = pose_r
    bundle["trans_r"] = trans_r
    bundle["shape_r"] = shape_r
    bundle["fitting_err_r"] = fitting_err_r
    bundle["rot_l"] = rot_l
    bundle["pose_l"] = pose_l
    bundle["trans_l"] = trans_l
    bundle["shape_l"] = shape_l
    bundle["fitting_err_l"] = fitting_err_l
    bundle["smplx_transl"] = smplx_transl
    bundle["smplx_global_orient"] = smplx_global_orient
    bundle["smplx_body_pose"] = smplx_body_pose
    bundle["smplx_jaw_pose"] = smplx_jaw_pose
    bundle["smplx_leye_pose"] = smplx_leye_pose
    bundle["smplx_reye_pose"] = smplx_reye_pose
    bundle["smplx_left_hand_pose"] = smplx_left_hand_pose
    bundle["smplx_right_hand_pose"] = smplx_right_hand_pose
    bundle["obj_params"] = obj_params
    bundle["obj_name"] = obj_name
    bundle["world2ego"] = world2ego
    bundle["K_ego"] = K_ego
    bundle["dist"] = dist

    dataset = PreprocessDataset(bundle)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=320,
        shuffle=False,
        num_workers=0
        # dataset, batch_size=320, shuffle=False, num_workers=8
    )
    return dataloader
