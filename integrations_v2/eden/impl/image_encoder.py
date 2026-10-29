# SPDX-FileCopyrightText: Copyright (c) 2026 Vast AI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for FlashDreams v0.2.0: the EdenLive fork patched
# flashdreams/infra/encoder/image/clip.py in place; the same preprocessing lives
# here as a subclass so the core CLIP encoder stays unchanged.

"""CLIP image encoder with the EdenLive (14B) cover preprocessing.

VAST's fork and its offline package hand the cover to ``CLIPImageProcessor`` as
uint8 HWC images and let the processor rescale them -- the PIL path of the
reference Wan I2V pipeline. The stock :class:`CLIPImageEncoder` passes float
``[0, 1]`` tensors with ``do_rescale=False`` instead, which resamples without
the uint8 clamp of bicubic overshoot. The fork also loads
``CLIPVisionModelWithProjection``, the class the checkpoint declares, so the
load consumes ``visual_projection.weight`` instead of reporting it unexpected
(features still come from ``hidden_states[-2]``). Only the 14B model uses CLIP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

import torch
from torch import Tensor

from flashdreams.core.io.hf import maybe_download_hf_repo_on_rank0
from flashdreams.infra.encoder import Encoder
from flashdreams.infra.encoder.image.clip import (
    CLIPImageEncoder,
    CLIPImageEncoderConfig,
)


@dataclass(kw_only=True)
class EdenCLIPImageEncoderConfig(CLIPImageEncoderConfig):
    """:class:`CLIPImageEncoderConfig` accepting a local EdenLive directory."""

    _target: type["EdenCLIPImageEncoder"] = field(
        default_factory=lambda: EdenCLIPImageEncoder
    )

    model_id_or_local_path: str  # type: ignore[assignment]
    """Local directory holding ``image_encoder/`` and ``image_processor/``."""


class EdenCLIPImageEncoder(CLIPImageEncoder):
    """CLIP image encoder fed uint8 covers, as the EdenLive reference does."""

    def __init__(self, config: EdenCLIPImageEncoderConfig) -> None:
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        Encoder.__init__(self, config)
        self.config = config
        maybe_download_hf_repo_on_rank0(
            config.model_id_or_local_path,
            allow_patterns=("image_encoder/**", "image_processor/**"),
        )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            config.model_id_or_local_path,
            subfolder="image_encoder",
            dtype=config.dtype,
            local_files_only=True,
        )
        self.image_encoder.eval().requires_grad_(False)
        self.image_processor = CLIPImageProcessor.from_pretrained(
            config.model_id_or_local_path,
            subfolder="image_processor",
            local_files_only=True,
        )

    @torch.no_grad()
    def forward(self, input: Tensor) -> Tensor:
        """Encode images ``[..., C, H, W]`` in ``[-1, 1]``; returns ``[..., 257, 1280]``."""
        from transformers import BatchFeature

        batch_shape = input.shape[:-3]
        images = input.reshape(math.prod(batch_shape), *input.shape[-3:])
        pixels = ((images.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        processed = cast(
            BatchFeature,
            self.image_processor(
                images=list(pixels.permute(0, 2, 3, 1).cpu().numpy()),
                return_tensors="pt",
            ),
        ).to(self.image_encoder.device, dtype=self.image_encoder.dtype)
        image_embeds = self.image_encoder(**processed, output_hidden_states=True)
        assert image_embeds.hidden_states is not None
        output: Tensor = image_embeds.hidden_states[-2]
        return output.reshape(*batch_shape, *output.shape[-2:])


__all__ = ["EdenCLIPImageEncoder", "EdenCLIPImageEncoderConfig"]
