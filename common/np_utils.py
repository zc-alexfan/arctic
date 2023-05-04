import numpy as np


def permute_np(x, idx):
    original_perm = tuple(range(len(x.shape)))
    x = np.moveaxis(x, original_perm, idx)
    return x
