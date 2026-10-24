# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan 2.1 14B DiT with the proxy-DINO channel-concat control head.

The AR teacher-forced recipe widens the patch embedding to
``16 noisy + 4 mask + 16 image latent + 1024 proxy-DINO = 1060`` input
channels and normalizes *only* the trailing DINO block with a trainable
affine RMSNorm immediately before the patch convolution. Parameters use the
DiT's BF16 storage; normalization statistics use FP32 inputs.

The patch embedding uses the reference's ``Conv3d`` on gathered input.
Its pre-conv normalization happens before patchify.
:meth:`normalize_control_channels` does exactly
that on the ``[..., T, C, H, W]`` control tensor, which keeps the numerics
identical to the reference (``permute`` channel-last, normalize in fp32,
restore dtype) while running once per AR chunk instead of once per
denoising step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetwork,
    WanDiTNetwork14BConfig,
    WanDiTNetworkCache,
)

# 16 noisy latent + 4 first-frame mask + 16 condition latent.
GAME14B_I2V_CHANNELS = 16 + 4 + 16
# Projected proxy-DINO feature width.
GAME14B_CONTROL_DIM = 1024


class Game14BDiTNetwork(WanDiTNetwork):
    """14B I2V DiT whose appended control channels carry their own RMSNorm.

    ``condition_norm`` is built in ``__init__`` (not lazily) because the
    checkpoint ships ``patch_embedding.condition_norm.weight`` and the
    strict state-dict load has to find the parameter already in place.
    """

    def __init__(self, config: "Game14BDiTNetworkConfig") -> None:
        super().__init__(config)
        self.control_dim = int(config.control_dim)
        if self.control_dim <= 0:
            raise ValueError("control_dim must be positive")
        if self.control_dim >= config.in_dim:
            raise ValueError(
                f"control_dim={self.control_dim} must be smaller than "
                f"in_dim={config.in_dim}"
            )
        # Registered under ``patch_embedding`` so the checkpoint key
        # ``patch_embedding.condition_norm.weight`` maps without a rename.
        # The transformer setup casts this affine to the DiT storage dtype.
        self.patch_embedding.condition_norm = nn.RMSNorm(
            self.control_dim,
            eps=config.eps,
            elementwise_affine=True,
            dtype=torch.float32,
        )

    def replace_text_embeddings(
        self,
        cache: WanDiTNetworkCache,
        text_embeddings: Tensor,
    ) -> None:
        """Swap the cached cross-attention text K/V for every block.

        Interface for per-chunk dynamic prompts. The Game2V rollout encodes one
        static prompt at startup and releases the text encoder; a rollout that
        wants per-chunk prompts reloads it once through
        ``Game14BInferencePipeline.encode_text`` and passes the embeddings here.

        Only the static cross-attention context changes; the self-attention
        KV cache is untouched, so the rollout horizon survives the swap.

        The write goes through :meth:`BlockKVCache.overwrite_kv_` rather than
        rebinding ``cross_attn.text``, for the reason spelled out on
        :meth:`Game5BDiTNetwork.replace_text_embeddings`: the forward reads this
        cache from inside a captured CUDA graph, so the storage address has to
        survive the swap. See that docstring for the full failure mode.
        """
        assert text_embeddings.shape[-2] == self.text_len, (
            f"text embeddings must have {self.text_len} tokens, got "
            f"{text_embeddings.shape[-2]}"
        )
        context_text = self.text_embedding(text_embeddings)
        for block, block_cache in zip(self.blocks, cache.block_caches):
            fresh = block.cross_attn.compute_kv(context_text)
            block_cache.cross_attn.text.overwrite_kv_(*fresh.clone_kv())

    def normalize_control_channels(self, control: Tensor) -> Tensor:
        """RMS-normalize a ``[..., T, C, H, W]`` control tensor over ``C``.

        Statistics are taken in fp32 over the channel axis and the input
        dtype is restored, matching the reference
        ``_RMSNormChannelConcatConv3d.forward``.
        """
        if control.shape[-3] != self.control_dim:
            raise ValueError(
                f"proxy-DINO control must have {self.control_dim} channels at "
                f"dim -3, got shape {tuple(control.shape)}"
            )
        # [..., T, C, H, W] -> [..., T, H, W, C] so C is the normalized dim.
        channel_last = control.movedim(-3, -1)
        normalized = self.patch_embedding.condition_norm(channel_last.float())
        return normalized.to(dtype=control.dtype).movedim(-1, -3)


@dataclass
class Game14BDiTNetworkConfig(WanDiTNetwork14BConfig):
    """Config for the proxy-DINO 14B network.

    ``in_dim`` covers the full channel-concat layout
    ``[noisy 16 | mask 4 | image latent 16 | proxy-DINO 1024]``; the
    checkpoint's ``patch_embedding.weight`` is ``[5120, 1060, 1, 2, 2]``,
    so the convolution must be constructed at this width before loading.
    """

    _target: type[Game14BDiTNetwork] = Game14BDiTNetwork

    in_dim: int = GAME14B_I2V_CHANNELS + GAME14B_CONTROL_DIM
    control_dim: int = GAME14B_CONTROL_DIM
    # Wan 2.1 I2V ships a CLIP vision branch (k_img / v_img / img_emb).
    cross_attn_enable_img: bool = True
    patch_embedding_type: str = "conv3d"
    # 40 heads shard evenly across 8 ranks, and the recipe's own launch uses
    # Ulysses head sharding.
    cp_method: str = "ulysses"


__all__ = [
    "GAME14B_CONTROL_DIM",
    "GAME14B_I2V_CHANNELS",
    "Game14BDiTNetwork",
    "Game14BDiTNetworkConfig",
]
