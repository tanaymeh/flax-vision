from itertools import repeat
import collections.abc


def _ntuple(n):
    """
    Taken from: https://github.com/rwightman/pytorch-image-models/blob/7c4682dc08e3964bc6eb2479152c5cdde465a961/timm/models/layers/helpers.py
    """

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)
