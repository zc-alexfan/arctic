import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common.ld_utils import unsort as unsort_list


# pytorch implementation for np.nanmean
# https://github.com/pytorch/pytorch/issues/21987#issuecomment-539402619
def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


def grad_norm(model):
    # compute norm of gradient for a model
    total_norm = None
    for p in model.parameters():
        if p.grad is not None:
            if total_norm is None:
                total_norm = 0
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2

    if total_norm is not None:
        total_norm = total_norm ** (1.0 / 2)
    else:
        total_norm = 0.0
    return total_norm


def pad_tensor_list(v_list: list):
    dev = v_list[0].device
    num_meshes = len(v_list)
    num_dim = 1 if len(v_list[0].shape) == 1 else v_list[0].shape[1]
    v_len_list = []
    for verts in v_list:
        v_len_list.append(verts.shape[0])

    pad_len = max(v_len_list)
    dtype = v_list[0].dtype
    if num_dim == 1:
        padded_tensor = torch.zeros(num_meshes, pad_len, dtype=dtype)
    else:
        padded_tensor = torch.zeros(num_meshes, pad_len, num_dim, dtype=dtype)
    for idx, (verts, v_len) in enumerate(zip(v_list, v_len_list)):
        padded_tensor[idx, :v_len] = verts
    padded_tensor = padded_tensor.to(dev)
    v_len_list = torch.LongTensor(v_len_list).to(dev)
    return padded_tensor, v_len_list


def unpad_vtensor(
    vtensor: (torch.Tensor), lens: (torch.LongTensor, torch.cuda.LongTensor)
):
    tensors_list = []
    for verts, vlen in zip(vtensor, lens):
        tensors_list.append(verts[:vlen])
    return tensors_list


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N, D1, D2, ..].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N, D1, D2, .., Dk, #classes].
    """
    y = torch.eye(num_classes).float()
    return y[labels]


def unsort(ten, sort_idx):
    """
    Unsort a tensor of shape (N, *) using the sort_idx list(N).
    Return a tensor of the pre-sorting order in shape (N, *)
    """
    assert isinstance(ten, torch.Tensor)
    assert isinstance(sort_idx, list)
    assert ten.shape[0] == len(sort_idx)

    out_list = list(torch.chunk(ten, ten.size(0), dim=0))
    out_list = unsort_list(out_list, sort_idx)
    out_list = torch.cat(out_list, dim=0)
    return out_list


def all_comb(X, Y):
    """
    Returns all possible combinations of elements in X and Y.
    X: (n_x, d_x)
    Y: (n_y, d_y)
    Output: Z: (n_x*x_y, d_x+d_y)
    Example:
    X = tensor([[8, 8, 8],
                [7, 5, 9]])
    Y = tensor([[3, 8, 7, 7],
                [3, 7, 9, 9],
                [6, 4, 3, 7]])
    Z = tensor([[8, 8, 8, 3, 8, 7, 7],
                [8, 8, 8, 3, 7, 9, 9],
                [8, 8, 8, 6, 4, 3, 7],
                [7, 5, 9, 3, 8, 7, 7],
                [7, 5, 9, 3, 7, 9, 9],
                [7, 5, 9, 6, 4, 3, 7]])
    """
    assert len(X.size()) == 2
    assert len(Y.size()) == 2
    X1 = X.unsqueeze(1)
    Y1 = Y.unsqueeze(0)
    X2 = X1.repeat(1, Y.shape[0], 1)
    Y2 = Y1.repeat(X.shape[0], 1, 1)
    Z = torch.cat([X2, Y2], -1)
    Z = Z.view(-1, Z.shape[-1])
    return Z


def toggle_parameters(model, requires_grad):
    """
    Set all weights to requires_grad or not.
    """
    for param in model.parameters():
        param.requires_grad = requires_grad


def detach_tensor(ten):
    """This function move tensor to cpu and convert to numpy"""
    if isinstance(ten, torch.Tensor):
        return ten.cpu().detach().numpy()
    return ten


def count_model_parameters(model):
    """
    Return the amount of parameters that requries gradients.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def reset_all_seeds(seed):
    """Reset all seeds for reproduciability."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_activation(name):
    """This function return an activation constructor by name."""
    if name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "relu":
        return nn.ReLU()
    elif name == "selu":
        return nn.SELU()
    elif name == "relu6":
        return nn.ReLU6()
    elif name == "softplus":
        return nn.Softplus()
    elif name == "softshrink":
        return nn.Softshrink()
    else:
        print("Undefined activation: %s" % (name))
        assert False


def stack_ll_tensors(tensor_list_list):
    """
    Recursively stack a list of lists of lists .. whose elements are tensors with the same shape
    """
    if isinstance(tensor_list_list, torch.Tensor):
        return tensor_list_list
    assert isinstance(tensor_list_list, list)
    if isinstance(tensor_list_list[0], torch.Tensor):
        return torch.stack(tensor_list_list)

    stacked_tensor = []
    for tensor_list in tensor_list_list:
        stacked_tensor.append(stack_ll_tensors(tensor_list))
    stacked_tensor = torch.stack(stacked_tensor)
    return stacked_tensor


def get_optim(name):
    """This function return an optimizer constructor by name."""
    if name == "adam":
        return optim.Adam
    elif name == "rmsprop":
        return optim.RMSprop
    elif name == "sgd":
        return optim.SGD
    else:
        print("Undefined optim: %s" % (name))
        assert False


def decay_lr(optimizer, gamma):
    """
    Decay the learning rate by gamma
    """
    assert isinstance(gamma, float)
    assert 0 <= gamma and gamma <= 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] *= gamma
