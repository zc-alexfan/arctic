import torch
from torch.utils.data import DataLoader

from common.torch_utils import reset_all_seeds
from src.datasets.arctic_dataset import ArcticDataset
from src.datasets.arctic_dataset_eval import ArcticDatasetEval
from src.datasets.tempo_dataset import TempoDataset
from src.datasets.tempo_inference_dataset import TempoInferenceDataset
from src.datasets.tempo_inference_dataset_eval import TempoInferenceDatasetEval


def fetch_dataset_eval(args, seq=None):
    if args.method in ["arctic_sf"]:
        DATASET = ArcticDatasetEval
    elif args.method in ["field_sf"]:
        DATASET = ArcticDatasetEval
    elif args.method in ["arctic_lstm", "field_lstm"]:
        DATASET = TempoInferenceDatasetEval
    else:
        assert False
    if seq is not None:
        split = args.run_on
    ds = DATASET(args=args, split=split, seq=seq)
    return ds


def fetch_dataset_devel(args, is_train, seq=None):
    split = args.trainsplit if is_train else args.valsplit
    if args.method in ["arctic_sf"]:
        if is_train:
            DATASET = ArcticDataset
        else:
            DATASET = ArcticDataset
    elif args.method in ["field_sf"]:
        if is_train:
            DATASET = ArcticDataset
        else:
            DATASET = ArcticDataset
    elif args.method in ["field_lstm", "arctic_lstm"]:
        if is_train:
            DATASET = TempoDataset
        else:
            DATASET = TempoInferenceDataset
    else:
        assert False
    if seq is not None:
        split = args.run_on
    ds = DATASET(args=args, split=split, seq=seq)
    return ds


def collate_custom_fn(data_list):
    data = data_list[0]
    _inputs, _targets, _meta_info = data
    out_inputs = {}
    out_targets = {}
    out_meta_info = {}

    for key in _inputs.keys():
        out_inputs[key] = []

    for key in _targets.keys():
        out_targets[key] = []

    for key in _meta_info.keys():
        out_meta_info[key] = []

    for data in data_list:
        inputs, targets, meta_info = data
        for key, val in inputs.items():
            out_inputs[key].append(val)

        for key, val in targets.items():
            out_targets[key].append(val)

        for key, val in meta_info.items():
            out_meta_info[key].append(val)

    for key in _inputs.keys():
        out_inputs[key] = torch.cat(out_inputs[key], dim=0)

    for key in _targets.keys():
        out_targets[key] = torch.cat(out_targets[key], dim=0)

    for key in _meta_info.keys():
        if key not in ["imgname", "query_names"]:
            out_meta_info[key] = torch.cat(out_meta_info[key], dim=0)
        else:
            out_meta_info[key] = sum(out_meta_info[key], [])

    return out_inputs, out_targets, out_meta_info


def fetch_dataloader(args, mode, seq=None):
    if mode == "train":
        reset_all_seeds(args.seed)
        dataset = fetch_dataset_devel(args, is_train=True)
        if type(dataset) == ArcticDataset:
            collate_fn = None
        else:
            collate_fn = collate_custom_fn
        return DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            shuffle=args.shuffle_train,
            collate_fn=collate_fn,
        )

    elif mode == "val" or mode == "eval":
        if "submit_" in args.extraction_mode:
            dataset = fetch_dataset_eval(args, seq=seq)
        else:
            dataset = fetch_dataset_devel(args, is_train=False, seq=seq)
        if type(dataset) in [ArcticDataset, ArcticDatasetEval]:
            collate_fn = None
        else:
            collate_fn = collate_custom_fn
        return DataLoader(
            dataset=dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
    else:
        assert False


def fetch_model(args):
    if args.method in ["arctic_sf"]:
        from src.models.arctic_sf.wrapper import ArcticSFWrapper as Wrapper
    elif args.method in ["arctic_lstm"]:
        from src.models.arctic_lstm.wrapper import ArcticLSTMWrapper as Wrapper
    elif args.method in ["field_sf"]:
        from src.models.field_sf.wrapper import FieldSFWrapper as Wrapper
    elif args.method in ["field_lstm"]:
        from src.models.field_lstm.wrapper import FieldLSTMWrapper as Wrapper
    else:
        assert False, f"Invalid method ({args.method})"
    model = Wrapper(args)
    return model
