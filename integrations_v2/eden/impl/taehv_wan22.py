# SPDX-FileCopyrightText: Copyright (c) 2026 Vast AI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for FlashDreams v0.2.0: the EdenLive fork added this decoder to
# flashdreams/recipes/taehv/__init__.py as TeahvVAEWan22Decoder; it lives in the
# integration here, renamed to match v0.2.0's Taehv* classes.

"""LightTAE decoder for Wan 2.2's 48-channel, 16x-spatial latents.

A drop-in stand-in for :class:`Wan22TI2V5BVAEDecoder` with the same compression
ratios (temporal 4, spatial 16) and the same normalized-latent input, so the 5B
pipeline swaps one for the other without touching frame counts or latent sizes.
The architecture is the :class:`~flashdreams.recipes.taehv.impl.TAEHV`
``wan22`` branch (``latent_channels=48``, ``patch_size=2``); the weights are
``decoder-5b/lighttaew2_2.pth`` from the EdenLive release (LightX2V
Autoencoders, Apache-2.0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

import torch
import tyro

from flashdreams.infra.decoder import StreamingVideoDecoder
from flashdreams.recipes.taehv import (
    TaehvVAEDecoder,
    TaehvVAEDecoderConfig,
    lighttae_state_dict_transform,
)
from flashdreams.recipes.taehv.checkpoint import StateDictTransform
from flashdreams.recipes.taehv.impl import TAEHV
from flashdreams.recipes.wan.autoencoder.vae import (
    _WAN22_TI2V_5B_LATENT_MEAN,
    _WAN22_TI2V_5B_LATENT_STD,
)


@dataclass(kw_only=True)
class TaehvVAEWan22DecoderConfig(TaehvVAEDecoderConfig):
    """Config for the Wan 2.2 (48-channel, 16x-spatial) LightTAE decoder."""

    _target: Annotated[type, tyro.conf.Suppress] = field(
        default_factory=lambda: TaehvVAEWan22Decoder
    )

    checkpoint_path: str
    """Path to the ``lighttaew2_2`` weights (no public default: the EdenLive
    configs resolve it against their model root)."""

    state_dict_transform: StateDictTransform | None = lighttae_state_dict_transform
    """Same flat-key remap and ``TGrow`` clip as the 16-channel ``lighttae``."""


class TaehvVAEWan22Decoder(TaehvVAEDecoder):
    """LightTAE decoder for the Wan 2.2 TI2V-5B latent space."""

    TEMPORAL_COMPRESSION_RATIO = 4
    # ``TAEHV.SPATIAL_COMPRESSION_RATIO`` counts the three 2x upsamples only;
    # the ``wan22`` branch's ``patch_size=2`` pixel shuffle adds the fourth.
    SPATIAL_COMPRESSION_RATIO = 16

    # The tiny decoder is trained on the VAE's *raw* latents, so the pipeline's
    # normalized latents are de-normalized with the Wan 2.2 VAE's own constants.
    _LIGHTTAE_MEAN: tuple[float, ...] = tuple(_WAN22_TI2V_5B_LATENT_MEAN)
    _LIGHTTAE_STD: tuple[float, ...] = tuple(_WAN22_TI2V_5B_LATENT_STD)

    def __init__(self, config: TaehvVAEWan22DecoderConfig) -> None:
        StreamingVideoDecoder.__init__(self, config)
        self.config: TaehvVAEWan22DecoderConfig = config
        self.need_scaled = True
        self.taehv = TAEHV(
            checkpoint_path=config.checkpoint_path,
            model_type="wan22",
            use_cuda_graph=config.use_cuda_graph,
            use_compile=config.use_compile,
            state_dict_transform=config.state_dict_transform,
        ).to(dtype=config.dtype)
        for name, values in (
            ("mean", self._LIGHTTAE_MEAN),
            ("std", self._LIGHTTAE_STD),
        ):
            self.register_buffer(
                name,
                torch.tensor(values, dtype=config.dtype).view(1, 1, -1, 1, 1),
                persistent=False,
            )


__all__ = ["TaehvVAEWan22Decoder", "TaehvVAEWan22DecoderConfig"]
