import fname
import re
from itertools import repeat
import collections.abc

__all__ = ["to_2tuple", "register_model", "list_models"]

model_dict = {}


def _ntuple(n):
    """
    Taken from: https://github.com/rwightman/pytorch-image-models/blob/7c4682dc08e3964bc6eb2479152c5cdde465a961/timm/models/layers/helpers.py
    """

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


def register_model(fn):
    name = ("_", "-", fn.__name__.lower())
    model_dict[name] = fn
    return fn


def list_models():
    pass


to_2tuple = _ntuple(2)
