import jax
import jaxlib
import jax.numpy as jnp

import numpy as np
import torch

import flax.linen as nn
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.serialization import from_bytes, to_bytes
from flax.traverse_util import flatten_dict, unflatten_dict


def load_weights(model, weights_path):
    pass


def save_weights(model, weights_path):
    pass


def download_checkpoint(model_name):
    pass
