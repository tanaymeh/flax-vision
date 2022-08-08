from tabnanny import check
import jax
import jaxlib
import jax.numpy as jnp
from jaxtyping import AbstractArray

from typing import Union, Optional, Callable

import flax
import flax.linen as nn

from utils.general import to_2tuple, register_model
from utils.decorators import add_start_doctring
from utils.serialization import (
    load_weights,
    download_checkpoint,
)
from utils.model_utils import drop_path, avg_pool, batch_norm

__all__ = [
    "poolformer_s12",
    "poolformer_s24",
    "poolformer_s36",
    "poolformer_m36",
    "poolformer_m48",
]


POOLFORMER_DOCSTRING = r"""
    TO BE IMPLEMENTED
"""


class DropPathModule(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    drop_prob: float

    @nn.compact
    def __call__(self, x: AbstractArray, key: AbstractArray) -> AbstractArray:
        return drop_path(x, key, self.drop_prob)


class IdentityModule(nn.Module):
    """Identity module."""

    def __call__(self, x: AbstractArray) -> AbstractArray:
        return x

    def __repr__(self):
        return "Identity()"


class PatchEmbeddingModule(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    patch_size: Optional[tuple] = to_2tuple(16)
    stride: Optional[tuple] = to_2tuple(16)
    padding: Optional[tuple] = to_2tuple(0)
    embed_dim: Optional[int] = 768
    norm_layer: Optional[Callable] = None

    @nn.compact
    def __call__(self, x: AbstractArray) -> AbstractArray:
        self.proj = nn.Conv(
            features=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.stride,
            padding=self.padding,
        )
        self.norm = (
            self.norm_layer(self.hidden_size) if self.norm_layer else IdentityModule
        )

        x = self.proj(x)
        x = self.norm(x)
        return x


class GroupNormModule(nn.GroupNorm):
    """
    1-Group GroupNorm for PoolFormer model.
    """

    def setup(self, num_channels):
        super().setup(num_channels)
        self.num_channels = num_channels

    @nn.compact
    def __call__(self, x: AbstractArray) -> AbstractArray:
        return nn.GroupNorm(self.num_channels, 1)(x)


class PoolingModule(nn.Module):
    pool_size: Optional[int] = to_2tuple(16)

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

    in_features: AbstractArray
    hidden_features: Optional[AbstractArray] = None
    out_features: Optional[AbstractArray] = None
    act_layer: Optional[Callable] = nn.gelu
    drop: float = 0.0

    @nn.compact
    def __call__(self, x: AbstractArray) -> AbstractArray:
        out_features = self.out_features or self.in_features
        hidden_features = self.hidden_features or self.in_features

        self.fc1 = nn.Conv(self.hidden_features, kernel_size=1)
        self.fc2 = nn.Conv(self.out_features, kernel_size=1)
        self.act = self.act_layer
        self.drop = nn.Dropout(self.drop)

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
        norm_layer=GroupNormModule,
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
    norm_layer=GroupNormModule,
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
    Class that builds the PoolFormer model from the Blocks
    """

    def setup(
        self,
        layers: list,
        embed_dims: Optional[tuple] = None,
        mlp_ratios: Optional[list] = None,
        downsamples: Optional[list] = None,
        pool_size: Optional[int] = 3,
        norm_layer: Optional[Callable] = GroupNormModule,
        act_layer: Optional[Callable] = nn.gelu,
        num_classes: Optional[int] = 1000,
        in_patch_size: Optional[int] = 7,
        in_stride: Optional[int] = 4,
        in_pad: Optional[int] = 2,
        down_patch_size: Optional[int] = 3,
        down_stride: Optional[int] = 2,
        down_pad: Optional[int] = 1,
        drop_rate: Optional[float] = 0.0,
        drop_path_rate: Optional[float] = 0.0,
        use_layer_scale: Optional[bool] = True,
        layer_scale_init_value: Optional[float] = 1e-5,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.patch_embed = PatchEmbeddingModule(
            patch_size=in_patch_size,
            stride=in_stride,
            padding=in_pad,
            embed_dim=embed_dims[0],
        )

        # Main Block
        self.network = []
        for i in range(len(layers)):
            stage = basic_blocks(
                embed_dims[i],
                i,
                layers,
                pool_size=pool_size,
                mlp_ratio=mlp_ratios[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            )
            self.network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                self.network.append(
                    PatchEmbeddingModule(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        padding=down_pad,
                        embed_dim=embed_dims[i + 1],
                    )
                )
        # Classifier Head
        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Dense(self.num_classes)

    def _forward_embeddings(self, x: AbstractArray) -> AbstractArray:
        return self.patch_embed(x)

    def __call__(self, x: AbstractArray) -> AbstractArray:
        x = self._forward_embeddings(x)
        for idx, block in enumerate(self.network):
            x = block(x)

        x = self.norm(x)
        x = self.head(x)
        return x

    def _init_weights(self):
        pass


@register_model
def poolformer_s12(
    num_classes: Optional[int] = 1000,
    dropout: Optional[float] = 0.1,
    pretrained: Optional[bool] = False,
    **kwargs,
):
    model_name = "poolformer_s12"
    layers = ([2, 2, 6, 2],)
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers,
        embed_dims=embed_dims,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        num_classes=num_classes,
        drop_rate=dropout,
        **kwargs,
    )
    if pretrained:
        checkpoint_loc = download_checkpoint(model_name)
        params = load_weights(checkpoint_loc)
        return model, params
    
    return model