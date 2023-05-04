import src.callbacks.process.process_arctic as process_arctic
import src.callbacks.process.process_generic as generic


def process_data(models, inputs, targets, meta_info, mode, args):
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
    meta_info["cams0"] = cams
    meta_info["parts_idx"] = pidx
    meta_info["v0.r.full"] = v0_r_full
    meta_info["v0.l.full"] = v0_l_full
    meta_info["v0.o.full"] = v0_o_full
    meta_info["mask"] = mask

    inputs, targets, meta_info = process_arctic.process_data(
        models, inputs, targets, meta_info, mode, args, field_max=args.max_dist
    )

    return inputs, targets, meta_info
