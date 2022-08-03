from json import detect_encoding
import jax
import jaxlib
import jax.numpy as jnp
from jaxtyping import AbstractArray, AbstractDtype

import flax.linen as nn


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


def batch_norm(
    inputs: AbstractArray,
    train: bool = True,
    epsilon: float = 1e-05,
    momentum: float = 0.99,
    params: dict = None,
    dtype: str = "float32",
):
    """
    Computes Batch Norm with optional parameters
    """

    if params is None:
        out = nn.BatchNorm(
            epsilon=epsilon,
            momentum=momentum,
            use_running_average=not train,
            dtype=dtype,
        )(inputs)
    else:
        out = nn.BatchNorm(
            epsilon=epsilon,
            momentum=momentum,
            bias_init=lambda *_: jnp.array(params["bias"]),
            scale_init=lambda *_: jnp.array(params["scale"]),
            mean_init=lambda *_: jnp.array(params["mean"]),
            var_init=lambda *_: jnp.array(params["var"]),
            use_running_average=not train,
            dtype=dtype,
        )(inputs)

    return out
