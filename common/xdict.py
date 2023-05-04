import numpy as np
import torch

import common.thing as thing


def _print_stat(key, thing):
    """
    Helper function for printing statistics about a key-value pair in an xdict.
    """
    mytype = type(thing)
    if isinstance(thing, (list, tuple)):
        print("{:<20}: {:<30}\t{:}".format(key, len(thing), mytype))
    elif isinstance(thing, (torch.Tensor)):
        dev = thing.device
        shape = str(thing.shape).replace(" ", "")
        print("{:<20}: {:<30}\t{:}\t{}".format(key, shape, mytype, dev))
    elif isinstance(thing, (np.ndarray)):
        dev = ""
        shape = str(thing.shape).replace(" ", "")
        print("{:<20}: {:<30}\t{:}".format(key, shape, mytype))
    else:
        print("{:<20}: {:}".format(key, mytype))


class xdict(dict):
    """
    A subclass of Python's built-in dict class, which provides additional methods for manipulating and operating on dictionaries.
    """

    def __init__(self, mydict=None):
        """
        Constructor for the xdict class. Creates a new xdict object and optionally initializes it with key-value pairs from the provided dictionary mydict. If mydict is not provided, an empty xdict is created.
        """
        if mydict is None:
            return

        for k, v in mydict.items():
            super().__setitem__(k, v)

    def subset(self, keys):
        """
        Returns a new xdict object containing only the key-value pairs with keys in the provided list 'keys'.
        """
        out_dict = {}
        for k in keys:
            out_dict[k] = self[k]
        return xdict(out_dict)

    def __setitem__(self, key, val):
        """
        Overrides the dict.__setitem__ method to raise an assertion error if a key already exists.
        """
        assert key not in self.keys(), f"Key already exists {key}"
        super().__setitem__(key, val)

    def search(self, keyword, replace_to=None):
        """
        Returns a new xdict object containing only the key-value pairs with keys that contain the provided keyword.
        """
        out_dict = {}
        for k in self.keys():
            if keyword in k:
                if replace_to is None:
                    out_dict[k] = self[k]
                else:
                    out_dict[k.replace(keyword, replace_to)] = self[k]
        return xdict(out_dict)

    def rm(self, keyword, keep_list=[], verbose=False):
        """
        Returns a new xdict object with keys that contain keyword removed. Keys in keep_list are excluded from the removal.
        """
        out_dict = {}
        for k in self.keys():
            if keyword not in k or k in keep_list:
                out_dict[k] = self[k]
            else:
                if verbose:
                    print(f"Removing: {k}")
        return xdict(out_dict)

    def overwrite(self, k, v):
        """
        The original assignment operation of Python dict
        """
        super().__setitem__(k, v)

    def merge(self, dict2):
        """
        Same as dict.update(), but raises an assertion error if there are duplicate keys between the two dictionaries.

        Args:
            dict2 (dict or xdict): The dictionary or xdict instance to merge with.

        Raises:
            AssertionError: If dict2 is not a dictionary or xdict instance.
            AssertionError: If there are duplicate keys between the two instances.
        """
        assert isinstance(dict2, (dict, xdict))
        mykeys = set(self.keys())
        intersect = mykeys.intersection(set(dict2.keys()))
        assert len(intersect) == 0, f"Merge failed: duplicate keys ({intersect})"
        self.update(dict2)

    def mul(self, scalar):
        """
        Multiplies each value (could be tensor, np.array, list) in the xdict instance by the provided scalar.

        Args:
            scalar (float): The scalar to multiply the values by.

        Raises:
            AssertionError: If scalar is not a float.
        """
        if isinstance(scalar, int):
            scalar = 1.0 * scalar
        assert isinstance(scalar, float)
        out_dict = {}
        for k in self.keys():
            if isinstance(self[k], list):
                out_dict[k] = [v * scalar for v in self[k]]
            else:
                out_dict[k] = self[k] * scalar
        return xdict(out_dict)

    def prefix(self, text):
        """
        Adds a prefix to each key in the xdict instance.

        Args:
            text (str): The prefix to add.

        Returns:
            xdict: The xdict instance with the added prefix.
        """
        out_dict = {}
        for k in self.keys():
            out_dict[text + k] = self[k]
        return xdict(out_dict)

    def replace_keys(self, str_src, str_tar):
        """
        Replaces a substring in all keys of the xdict instance.

        Args:
            str_src (str): The substring to replace.
            str_tar (str): The replacement string.

        Returns:
            xdict: The xdict instance with the replaced keys.
        """
        out_dict = {}
        for k in self.keys():
            old_key = k
            new_key = old_key.replace(str_src, str_tar)
            out_dict[new_key] = self[k]
        return xdict(out_dict)

    def postfix(self, text):
        """
        Adds a postfix to each key in the xdict instance.

        Args:
            text (str): The postfix to add.

        Returns:
            xdict: The xdict instance with the added postfix.
        """
        out_dict = {}
        for k in self.keys():
            out_dict[k + text] = self[k]
        return xdict(out_dict)

    def sorted_keys(self):
        """
        Returns a sorted list of the keys in the xdict instance.

        Returns:
            list: A sorted list of keys in the xdict instance.
        """
        return sorted(list(self.keys()))

    def to(self, dev):
        """
        Moves the xdict instance to a specific device.

        Args:
            dev (torch.device): The device to move the instance to.

        Returns:
            xdict: The xdict instance moved to the specified device.
        """
        if dev is None:
            return self
        raw_dict = dict(self)
        return xdict(thing.thing2dev(raw_dict, dev))

    def to_torch(self):
        """
        Converts elements in the xdict to Torch tensors and returns a new xdict.

        Returns:
        xdict: A new xdict with Torch tensors as values.
        """
        return xdict(thing.thing2torch(self))

    def to_np(self):
        """
        Converts elements in the xdict to numpy arrays and returns a new xdict.

        Returns:
        xdict: A new xdict with numpy arrays as values.
        """
        return xdict(thing.thing2np(self))

    def tolist(self):
        """
        Converts elements in the xdict to Python lists and returns a new xdict.

        Returns:
        xdict: A new xdict with Python lists as values.
        """
        return xdict(thing.thing2list(self))

    def print_stat(self):
        """
        Prints statistics for each item in the xdict.
        """
        for k, v in self.items():
            _print_stat(k, v)

    def detach(self):
        """
        Detaches all Torch tensors in the xdict from the computational graph and moves them to the CPU.
        Non-tensor objects are ignored.

        Returns:
        xdict: A new xdict with detached Torch tensors as values.
        """
        return xdict(thing.detach_thing(self))

    def has_invalid(self):
        """
        Checks if any of the Torch tensors in the xdict contain NaN or Inf values.

        Returns:
        bool: True if at least one tensor contains NaN or Inf values, False otherwise.
        """
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                if torch.isnan(v).any():
                    print(f"{k} contains nan values")
                    return True
                if torch.isinf(v).any():
                    print(f"{k} contains inf values")
                    return True
        return False

    def apply(self, operation, criterion=None):
        """
        Applies an operation to the values in the xdict, based on an optional criterion.

        Args:
        operation (callable): A callable object that takes a single argument and returns a value.
        criterion (callable, optional): A callable object that takes two arguments (key and value) and returns a boolean.

        Returns:
        xdict: A new xdict with the same keys as the original, but with the values modified by the operation.
        """
        out = {}
        for k, v in self.items():
            if criterion is None or criterion(k, v):
                out[k] = operation(v)
        return xdict(out)

    def save(self, path, dev=None, verbose=True):
        """
        Saves the xdict to disk as a Torch tensor.

        Args:
        path (str): The path to save the xdict.
        dev (torch.device, optional): The device to use for saving the tensor (default is CPU).
        verbose (bool, optional): Whether to print a message indicating that the xdict has been saved (default is True).
        """
        if verbose:
            print(f"Saving to {path}")
        torch.save(self.to(dev), path)
