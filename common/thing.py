import numpy as np
import torch

"""
This file stores functions for conversion between numpy and torch, torch, list, etc.
Also deal with general operations such as to(dev), detach, etc.
"""


def thing2list(thing):
    if isinstance(thing, torch.Tensor):
        return thing.tolist()
    if isinstance(thing, np.ndarray):
        return thing.tolist()
    if isinstance(thing, dict):
        return {k: thing2list(v) for k, v in md.items()}
    if isinstance(thing, list):
        return [thing2list(ten) for ten in thing]
    return thing


def thing2dev(thing, dev):
    if hasattr(thing, "to"):
        thing = thing.to(dev)
        return thing
    if isinstance(thing, list):
        return [thing2dev(ten, dev) for ten in thing]
    if isinstance(thing, tuple):
        return tuple(thing2dev(list(thing), dev))
    if isinstance(thing, dict):
        return {k: thing2dev(v, dev) for k, v in thing.items()}
    if isinstance(thing, torch.Tensor):
        return thing.to(dev)
    return thing


def thing2np(thing):
    if isinstance(thing, list):
        return np.array(thing)
    if isinstance(thing, torch.Tensor):
        return thing.cpu().detach().numpy()
    if isinstance(thing, dict):
        return {k: thing2np(v) for k, v in thing.items()}
    return thing


def thing2torch(thing):
    if isinstance(thing, list):
        return torch.tensor(np.array(thing))
    if isinstance(thing, np.ndarray):
        return torch.from_numpy(thing)
    if isinstance(thing, dict):
        return {k: thing2torch(v) for k, v in thing.items()}
    return thing


def detach_thing(thing):
    if isinstance(thing, torch.Tensor):
        return thing.cpu().detach()
    if isinstance(thing, list):
        return [detach_thing(ten) for ten in thing]
    if isinstance(thing, tuple):
        return tuple(detach_thing(list(thing)))
    if isinstance(thing, dict):
        return {k: detach_thing(v) for k, v in thing.items()}
    return thing
