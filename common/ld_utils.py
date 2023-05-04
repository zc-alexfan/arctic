import itertools

import numpy as np
import torch


def sort_dict(disordered):
    sorted_dict = {k: disordered[k] for k in sorted(disordered)}
    return sorted_dict


def prefix_dict(mydict, prefix):
    out = {prefix + k: v for k, v in mydict.items()}
    return out


def postfix_dict(mydict, postfix):
    out = {k + postfix: v for k, v in mydict.items()}
    return out


def unsort(L, sort_idx):
    assert isinstance(sort_idx, list)
    assert isinstance(L, list)
    LL = zip(sort_idx, L)
    LL = sorted(LL, key=lambda x: x[0])
    _, L = zip(*LL)
    return list(L)


def cat_dl(out_list, dim, verbose=True, squeeze=True):
    out = {}
    for key, val in out_list.items():
        if isinstance(val[0], torch.Tensor):
            out[key] = torch.cat(val, dim=dim)
            if squeeze:
                out[key] = out[key].squeeze()
        elif isinstance(val[0], np.ndarray):
            out[key] = np.concatenate(val, axis=dim)
            if squeeze:
                out[key] = np.squeeze(out[key])
        elif isinstance(val[0], list):
            out[key] = sum(val, [])
        else:
            if verbose:
                print(f"Ignoring {key} undefined type {type(val[0])}")
    return out


def stack_dl(out_list, dim, verbose=True, squeeze=True):
    out = {}
    for key, val in out_list.items():
        if isinstance(val[0], torch.Tensor):
            out[key] = torch.stack(val, dim=dim)
            if squeeze:
                out[key] = out[key].squeeze()
        elif isinstance(val[0], np.ndarray):
            out[key] = np.stack(val, axis=dim)
            if squeeze:
                out[key] = np.squeeze(out[key])
        elif isinstance(val[0], list):
            out[key] = sum(val, [])
        else:
            out[key] = val
            if verbose:
                print(f"Processing {key} undefined type {type(val[0])}")
    return out


def add_prefix_postfix(mydict, prefix="", postfix=""):
    assert isinstance(mydict, dict)
    return dict((prefix + key + postfix, value) for (key, value) in mydict.items())


def ld2dl(LD):
    assert isinstance(LD, list)
    assert isinstance(LD[0], dict)
    """
    A list of dict (same keys) to a dict of lists
    """
    dict_list = {k: [dic[k] for dic in LD] for k in LD[0]}
    return dict_list


class NameSpace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def dict2ns(mydict):
    """
    Convert dict objec to namespace
    """
    return NameSpace(mydict)


def ld2dev(ld, dev):
    """
    Convert tensors in a list or dict to a device recursively
    """
    if isinstance(ld, torch.Tensor):
        return ld.to(dev)
    if isinstance(ld, dict):
        for k, v in ld.items():
            ld[k] = ld2dev(v, dev)
        return ld
    if isinstance(ld, list):
        return [ld2dev(x, dev) for x in ld]
    return ld


def all_comb_dict(hyper_dict):
    assert isinstance(hyper_dict, dict)
    keys, values = zip(*hyper_dict.items())
    permute_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return permute_dicts
