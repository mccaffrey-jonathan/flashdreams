# SPDX-FileCopyrightText: Copyright (c) 2026 Vast AI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for FlashDreams v0.2.0: the EdenLive fork patched
# flashdreams/recipes/wan/autoencoder/vae.py in place; the encoder-side changes
# live here as a subclass so the core VAE numerics stay unchanged.

"""Wan 2.2 TI2V-5B VAE encoder with the reference first-frame numerics.

The 5B pipeline runs the VAE encoder exactly once per rollout, on the cover
image whose latent is stamped into chunk 0, so that one latent is worth
matching the reference closely. Two differences from :class:`WanVAEEncoder`:

* ``RMS_norm`` normalises in fp32 and casts back, as Diffusers >= 0.38
  ``WanRMS_norm`` does for bf16 inputs.
* The latent ``inv_std`` is the reciprocal of the bf16-cast std (the
  reference's order of operations), not a bf16 cast of the fp32 reciprocal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Annotated

import torch
import torch.nn.functional as F
import tyro
from torch import Tensor

from flashdreams.recipes.wan.autoencoder.vae import (
    RMS_norm,
    Wan22TI2V5BVAEEncoderConfig,
    WanVAECache,
    WanVAEEncoder,
)


class FP32RMSNorm(RMS_norm):
    """:class:`RMS_norm` whose normalisation runs in fp32."""

    def forward(self, x: Tensor) -> Tensor:
        dim = 1 if self.channel_first else -1
        return (
            F.normalize(x.float(), dim=dim).to(x.dtype) * self.scale * self.gamma
            + self.bias
        )


@dataclass(kw_only=True)
class EdenWan22TI2V5BVAEEncoderConfig(Wan22TI2V5BVAEEncoderConfig):
    """:class:`Wan22TI2V5BVAEEncoderConfig` with the reference numerics."""

    _target: Annotated[type, tyro.conf.Suppress] = field(
        default_factory=lambda: EdenWanVAEEncoder
    )


class EdenWanVAEEncoder(WanVAEEncoder):
    """:class:`WanVAEEncoder` with fp32 RMS norms and a bf16-order ``inv_std``."""

    def __init__(self, config: EdenWan22TI2V5BVAEEncoderConfig) -> None:
        super().__init__(config)
        for module in self.vae.modules():
            if type(module) is RMS_norm:
                module.__class__ = FP32RMSNorm
        # Assigning to a registered buffer name updates the buffer, so it still
        # follows ``.to(device)``.
        self.vae.inv_std = (
            1.0 / torch.tensor(config.latent_std, dtype=config.dtype)
        ).view(1, config.z_dim, 1, 1, 1)

    @torch.no_grad()
    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: WanVAECache | None = None,
    ) -> Tensor:
        if cache is None:
            cache = self.initialize_autoregressive_cache()
        assert input.ndim >= 4, "Expected input to have shape [..., T, C, H, W]"
        *batch_shape, T, C, H, W = input.shape
        x = input.reshape(math.prod(batch_shape), T, C, H, W).to(
            dtype=self.config.dtype
        )
        z = self.vae.encode(x.transpose(1, 2).contiguous(), cache=cache).transpose(1, 2)
        return z.reshape(*batch_shape, *z.shape[1:])


__all__ = ["EdenWan22TI2V5BVAEEncoderConfig", "EdenWanVAEEncoder", "FP32RMSNorm"]
