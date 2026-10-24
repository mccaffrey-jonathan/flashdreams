# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Game2V wiring for the proxy-DINO conditioned 5B model.

The first frame, prompt and step-input hooks are the 14B's: both models read the
per-game cover and prompt out of ``games/registry.json``, take the encoded render
as their step input, and support ``--dynamic-prompt`` through the same
``encode_text`` / ``replace_text_embeddings`` pair. Only the control stream
differs, because the grid the features have to land on is set by the VAE:

* The 14B's Wan 2.1 VAE is 8x spatial, so at 480x832 the DiT grid is 60x104.
* This model's Wan 2.2 TI2V-5B VAE is 16x spatial, so at 704x1280 it is 44x80.

Everything else -- the DINOv3 backbone, the finetuned projection, the overlapped
side-stream encode and the rank-0-encodes-and-broadcasts topology -- is shared
with the 14B and imported from :mod:`eden.impl.game14b`.
"""

from __future__ import annotations

from typing import Any

from eden.impl.game14b.control_stream import ProxyDinoControlStream
from eden.impl.game14b.dino import ProxyDinoEncoderConfig

# The per-game cover/prompt resolution and the step-generate hook are
# model-independent, so the 14B's are reused verbatim rather than copied.
from eden.apps.game2v.game14b import (
    game_cover_path,
    generate_game14b_step as generate_game5b_step,
    resolve_game14b_conditioning as resolve_game5b_conditioning,
)

# Wan 2.2 TI2V-5B's VAE packs a 2x2 patchify on top of the usual 8x encoder
# downsample. The control rides on the pre-patchify latent grid (the DiT applies
# its own 2x2 patchify afterwards), which is pixels / 16.
GAME2V_5B_VAE_SPATIAL_RATIO = 16


def make_game5b_control_stream(
    config: Any, session_desc: Any
) -> ProxyDinoControlStream:
    """Build the overlapped proxy-DINO encoder for one 5B session.

    Latent dimensions come from the model's own frame size divided by the Wan
    2.2 TI2V-5B VAE's 16x spatial ratio, so the features land on the DiT's grid
    without a second resample at consume time.
    """
    from eden.game5b_config import GAME2V_5B_LEN_T, _SPEC
    from eden.pipeline_yaml import component_path

    pixel_width = config.model_video_width or session_desc.video_width
    pixel_height = config.model_video_height or session_desc.video_height
    if (
        pixel_height % GAME2V_5B_VAE_SPATIAL_RATIO
        or pixel_width % GAME2V_5B_VAE_SPATIAL_RATIO
    ):
        raise ValueError(
            f"Game2V 5B frame size ({pixel_height}, {pixel_width}) must be "
            f"divisible by the VAE's {GAME2V_5B_VAE_SPATIAL_RATIO}x spatial ratio."
        )
    return ProxyDinoControlStream(
        ProxyDinoEncoderConfig(
            backbone_repo=component_path(_SPEC, "vision_encoder"),
            projection_path=component_path(_SPEC, "vision_projection"),
            device=str(config.device),
            compile_encoder=getattr(config, "compile_control_encoder", True),
        ),
        latent_height=pixel_height // GAME2V_5B_VAE_SPATIAL_RATIO,
        latent_width=pixel_width // GAME2V_5B_VAE_SPATIAL_RATIO,
        len_t=GAME2V_5B_LEN_T,
        max_latent_frames=_SPEC["max_latent_frames"],
    )


__all__ = [
    "GAME2V_5B_VAE_SPATIAL_RATIO",
    "game_cover_path",
    "generate_game5b_step",
    "make_game5b_control_stream",
    "resolve_game5b_conditioning",
]
