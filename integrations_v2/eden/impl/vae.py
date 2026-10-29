# SPDX-FileCopyrightText: Copyright (c) 2026 Vast AI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for FlashDreams v0.2.0: the EdenLive fork patched
# flashdreams/recipes/wan/autoencoder/vae.py in place; the encoder-side changes
# live here as a subclass so the core VAE numerics stay unchanged.

"""Wan VAE encoder/decoder with the EdenLive reference numerics.

The pipelines run the VAE encoder once per rollout, on the cover whose latent
conditions chunk 0 (5B stamp, 14B channel concat), so that latent is worth
matching the reference closely. Differences from the stock Wan VAE:

* ``RMS_norm`` normalises in fp32 and casts back, as Diffusers >= 0.38
  ``WanRMS_norm`` does for bf16 inputs (encoder and full decoder).
* The encoder's latent ``inv_std`` is the reciprocal of the bf16-cast std (the
  reference's order of operations), not a bf16 cast of the fp32 reciprocal.
  The decoder keeps the fp32 reciprocal, as VAST's fork does.

The LightTAE presets decode with :mod:`eden.impl.taehv_wan22` instead.
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
    Wan22TI2V5BVAEDecoderConfig,
    Wan22TI2V5BVAEEncoderConfig,
    WanVAECache,
    WanVAEDecoder,
    WanVAEDecoderConfig,
    WanVAEEncoder,
    WanVAEEncoderConfig,
)


class FP32RMSNorm(RMS_norm):
    """:class:`RMS_norm` whose normalisation runs in fp32."""

    def forward(self, x: Tensor) -> Tensor:
        dim = 1 if self.channel_first else -1
        return (
            F.normalize(x.float(), dim=dim).to(x.dtype) * self.scale * self.gamma
            + self.bias
        )


def _use_fp32_rms_norms(module: torch.nn.Module) -> None:
    """Re-class every stock ``RMS_norm`` under ``module`` (no new state)."""
    for child in module.modules():
        if type(child) is RMS_norm:
            child.__class__ = FP32RMSNorm


@dataclass(kw_only=True)
class EdenWanVAEEncoderConfig(WanVAEEncoderConfig):
    """Wan 2.1 (14B) :class:`WanVAEEncoderConfig` with the reference numerics."""

    _target: Annotated[type, tyro.conf.Suppress] = field(
        default_factory=lambda: EdenWanVAEEncoder
    )


@dataclass(kw_only=True)
class EdenWan22TI2V5BVAEEncoderConfig(Wan22TI2V5BVAEEncoderConfig):
    """:class:`Wan22TI2V5BVAEEncoderConfig` with the reference numerics."""

    _target: Annotated[type, tyro.conf.Suppress] = field(
        default_factory=lambda: EdenWanVAEEncoder
    )


@dataclass(kw_only=True)
class EdenWanVAEDecoderConfig(WanVAEDecoderConfig):
    """Wan 2.1 (14B) :class:`WanVAEDecoderConfig` with fp32 RMS norms."""

    _target: Annotated[type, tyro.conf.Suppress] = field(
        default_factory=lambda: EdenWanVAEDecoder
    )


@dataclass(kw_only=True)
class EdenWan22TI2V5BVAEDecoderConfig(Wan22TI2V5BVAEDecoderConfig):
    """:class:`Wan22TI2V5BVAEDecoderConfig` with fp32 RMS norms."""

    _target: Annotated[type, tyro.conf.Suppress] = field(
        default_factory=lambda: EdenWanVAEDecoder
    )


class EdenWanVAEDecoder(WanVAEDecoder):
    """:class:`WanVAEDecoder` whose RMS norms normalise in fp32."""

    def __init__(self, config: WanVAEDecoderConfig) -> None:
        super().__init__(config)
        _use_fp32_rms_norms(self.vae)


class EdenWanVAEEncoder(WanVAEEncoder):
    """:class:`WanVAEEncoder` with fp32 RMS norms and a bf16-order ``inv_std``."""

    def __init__(self, config: WanVAEEncoderConfig) -> None:
        super().__init__(config)
        _use_fp32_rms_norms(self.vae)
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


__all__ = [
    "EdenWan22TI2V5BVAEDecoderConfig",
    "EdenWan22TI2V5BVAEEncoderConfig",
    "EdenWanVAEDecoder",
    "EdenWanVAEDecoderConfig",
    "EdenWanVAEEncoder",
    "EdenWanVAEEncoderConfig",
    "FP32RMSNorm",
]
