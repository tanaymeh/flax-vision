import jax
import jaxlib
import jax.numpy as jnp
from jaxtyping import AbstractArray

from typing import Union

import flax
import flax.linen as nn
from flax.serialization import to_bytes, from_bytes

from utils.decorators import add_start_doctring

POOLFORMER_DOCSTRING = r"""
    TO BE IMPLEMENTED
"""


def drop_path(x: AbstractArray, key, drop_prob: float = 0.0) -> AbstractArray:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0:
        return x
    keep_prob = 1 - drop_prob
    keep_shape = (x.shape[0], 1, 1, 1)
    keep_mask = keep_prob + jax.random.bernoulli(key, keep_shape)
    output = (x / keep_prob) * keep_mask
    return output


def avg_pool(inputs, window_shape, strides=None, padding="VALID"):
    """
    Pools the input by taking the average over a window.
    In comparison to nn.avg_pool(), this pooling operation does not
    consider the padded zero's for the average computation.
    """
    assert len(window_shape) == 2

    y = nn.pool(inputs, 0.0, jax.lax.add, window_shape, strides, padding)
    counts = nn.pool(
        jnp.ones_like(inputs), 0.0, jax.lax.add, window_shape, strides, padding
    )
    y = y / counts
    return y


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def setup(self, drop_prob: float = 0.0):
        self.drop_prob = drop_prob

    def call(self, x: AbstractArray, key: AbstractArray) -> AbstractArray:
        return drop_path(x, key, self.drop_prob)

    def __call__(self, x: AbstractArray, key: AbstractArray) -> AbstractArray:
        return self.call(x, key)

    def __repr__(self):
        return f"DropPath(drop_prob={self.drop_prob})"


class Identity(nn.Module):
    """Identity module."""

    def call(self, x: AbstractArray) -> AbstractArray:
        return x

    def __call__(self, x: AbstractArray) -> AbstractArray:
        return self.call(x)

    def __repr__(self):
        return "Identity()"


class PatchEmbeddings(nn.Module):
    """
    PatchEmbeddings for PoolFormer model.
    """

    def setup(
        self,
        hidden_size: int,
        num_channels: int,
        patch_size: tuple,
        stride: tuple,
        padding: tuple,
        norm_layer=None,
    ):
        self.hidden_size = hidden_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.norm_layer = norm_layer

        self.projection = nn.Conv(
            features=self.hidden_size,
            kernel_size=self.patch_size,
            strides=self.stride,
            padding=self.padding,
        )
        self.norm = norm_layer(self.hidden_size) if norm_layer else Identity

    def __call__(self, x: AbstractArray) -> AbstractArray:
        x = self.projection(x)
        x = self.norm(x)
        return x


class SingleGroupNorm(nn.GroupNorm):
    """
    1-Group GroupNorm for PoolFormer model.
    """

    def setup(self, num_channels: int):
        self.num_channels = num_channels

    def call(self, x: AbstractArray) -> AbstractArray:
        return nn.GroupNorm(self.num_channels, 1)(x)

    def __call__(self, x: AbstractArray) -> AbstractArray:
        return self.call(x)

    def __repr__(self):
        return f"GroupNorm(num_channels={self.num_channels})"


class Pooling(nn.Module):
    def setup(self, pool_size):
        self.pool_size = pool_size

    def __call__(self, inputs: AbstractArray) -> AbstractArray:
        pooled_output = avg_pool(
            inputs, self.pool_size, strides=1, padding=self.pool_size // 2
        )
        return pooled_output - inputs


class MLP(nn.Module):
    """
    Implementation of MLP with 1x1 Convolution block
    Input: [batch_size, num_channels, height, width]
    """

    def setup(
        self,
        in_features: AbstractArray,
        hidden_features=None,
        out_features=None,
        act_layer=nn.gelu,
        drop=0.0,
    ):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
    
    def __call__(self, inputs: AbstractArray) -> AbstractArray:
        pass