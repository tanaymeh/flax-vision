import jax
import jaxlib
import jax.numpy as jnp
from jaxtyping import AbstractArray

from typing import Union

import flax
import flax.linen as nn
from flax.serialization import to_bytes, from_bytes

from utils.decorators import add_start_doctring
from utils.general import to_2tuple
from utils.model_utils import drop_path, avg_pool, batch_norm

POOLFORMER_DOCSTRING = r"""
    TO BE IMPLEMENTED
"""


class DropPathModule(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def setup(self, drop_prob: float = 0.0):
        self.drop_prob = drop_prob

    def call(self, x: AbstractArray, key: AbstractArray) -> AbstractArray:
        return drop_path(x, key, self.drop_prob)

    def __call__(self, x: AbstractArray, key: AbstractArray) -> AbstractArray:
        return self.call(x, key)

    def __repr__(self):
        return f"DropPath(drop_prob={self.drop_prob})"


class IdentityModule(nn.Module):
    """Identity module."""

    def call(self, x: AbstractArray) -> AbstractArray:
        return x

    def __call__(self, x: AbstractArray) -> AbstractArray:
        return self.call(x)

    def __repr__(self):
        return "Identity()"


class PatchEmbeddingModule(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def setup(
        self, patch_size=16, stride=16, padding=0, embed_dim=768, norm_layer=None
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)

        self.proj = nn.Conv(
            features=embed_dim,
            kernel_size=self.patch_size,
            strides=self.stride,
            padding=self.padding,
        )
        self.norm = norm_layer(self.hidden_size) if norm_layer else IdentityModule

    def __call__(self, x: AbstractArray) -> AbstractArray:
        x = self.proj(x)
        x = self.norm(x)
        return x


class SingleGroupNormModule(nn.GroupNorm):
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


class PoolingModule(nn.Module):
    def setup(self, pool_size):
        self.pool_size = pool_size

    def __call__(self, inputs: AbstractArray) -> AbstractArray:
        pooled_output = avg_pool(
            inputs, self.pool_size, strides=1, padding=self.pool_size // 2
        )
        return pooled_output - inputs


class MLPModule(nn.Module):
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

        self.fc1 = nn.Conv(hidden_features, kernel_size=1)
        self.fc2 = nn.Conv(out_features, kernel_size=1)
        self.act = act_layer
        self.drop = nn.Dropout(drop)

    def __call__(self, x: AbstractArray) -> AbstractArray:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoolFormerBlockModule(nn.Module):
    """
    Implementation of One PoolFormer Block
    """

    def __init__(
        self,
        dim,
        pool_size=3,
        mlp_ratio=4.0,
        act_layer=nn.gelu,
        norm_layer=SingleGroupNormModule,
        drop=0.0,
        drop_path=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.token_mixer = PoolingModule(pool_size=pool_size)
        mlp_hidden_features = int(dim * mlp_ratio)

        self.mlp = MLPModule(
            in_features=dim,
            hidden_features=mlp_hidden_features,
            act_layer=act_layer,
            drop=drop,
        )

        self.drop_path = (
            DropPathModule(drop_prob=drop_path) if drop_path > 0.0 else IdentityModule()
        )

        self.use_layer_scale = use_layer_scale
        if self.use_layer_scale:
            self.layer_scale_1 = nn.Dense(
                layer_scale_init_value * jnp.ones((dim)), use_bias=False
            )
            self.layer_scale_2 = nn.Dense(
                layer_scale_init_value * jnp.ones((dim)), use_bias=False
            )

    def __call__(self, x: AbstractArray) -> AbstractArray:
        # Haven't figured out how to expand dimensions of a Flax Layer so training is not available
        x = x + self.drop_path(self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


def basic_blocks(
    dim,
    index,
    layers,
    pool_size=3,
    mlp_ratio=4.0,
    act_layer=nn.gelu,
    norm_layer=SingleGroupNormModule,
    drop_rate=0.0,
    drop_path_ratio=0.0,
    use_layer_scale=True,
    layer_scale_init_value=1e-5,
):
    """
    Generate PoolFormer blocks for a single stage
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = (
            drop_path_ratio * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        )
        blocks.append(
            PoolFormerBlockModule(
                dim,
                pool_size=pool_size,
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop=drop_rate,
                drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            )
        )
    blocks = nn.Sequential(*blocks)
    return blocks


class PoolFormer(nn.Module):
    """
    Main class for the model
    """

    def setup(
        self,
        layers,
        embed_dims=None,
        mlp_ratio=None,
        downsamples=None,
        pool_size=3,
        norm_layer=SingleGroupNormModule,
        act_layer=nn.gelu,
        num_classes=1000,
        in_patch_size=7,
        in_stride=4,
        in_pad=2,
        down_patch_size=3,
        down_stride=2,
        down_pad=1,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        fork_feat=False,
        init_cfg=None,
        pretrained=None,
        **kwargs,
    ):
        super().__init__()
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = PatchEmbeddingModule(
            patch_size=16, stride=16, padding=0, embed_dim=768, norm_layer=None
        )
