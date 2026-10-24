# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config for the proxy-DINO conditioned AR 14B Game2V model.

The values live in :file:`configs/game2v_14b.yaml`, which this module loads and
maps onto the dataclasses.

Kept out of :mod:`eden.config` because that module's ``PIPELINE_CONFIGS`` is
typed to the camera-control pipeline; this model conditions on rendered game
frames through DINOv3 instead of Plücker camera volumes, so it is a different
pipeline type entirely.

A 4-step distilled AR rollout: chunk 3 at 480x832, a shift-3 schedule yielding
timesteps ``[1000, 900, 750, 500]``, CFG off, and an 18-frame attention budget
split as 6 sink + 12 rolling.
"""

from __future__ import annotations

from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler import FlowMatchSchedulerConfig
from flashdreams.infra.encoder.image.clip import CLIPImageEncoderConfig
from flashdreams.infra.encoder.text.umt5 import UMT5TextEncoderConfig
from flashdreams.recipes.taehv import TeahvVAEDecoderConfig
from flashdreams.recipes.wan.autoencoder.vae import (
    WanVAEDecoderConfig,
    WanVAEEncoderConfig,
)
from eden.impl.game14b.encoder import Game14BCtrlEncoderConfig
from eden.impl.game14b.network import Game14BDiTNetworkConfig
from eden.impl.game14b.pipeline import Game14BInferencePipelineConfig
from eden.impl.game14b.transformer import Game14BTransformerConfig
from eden.pipeline_yaml import (
    component_path,
    load_pipeline_yaml,
    resolve_path,
    scheduler_kwargs,
)

# Every number below comes from configs/game2v_14b.yaml; the provenance of each
# one is documented there. This module only maps them onto the dataclasses.
_SPEC = load_pipeline_yaml("game2v_14b")

GAME2V_EDEN_WORLD_V2_14B_CHECKPOINT_PATH = component_path(_SPEC, "transformer")
"""The distilled 14B DiT, published as a single safetensors file with Diffusers
keys. Read through
:func:`eden.impl.game14b.checkpoint.load_game14b_dit_state_dict`, which applies
the shared Diffusers remap plus the CLIP-branch and control-norm keys.
"""

GAME2V_LEN_T = _SPEC["len_t"]
GAME2V_WINDOW_SIZE_T = _SPEC["window_size_t"]

PIPELINE_GAME2V_EDEN_WORLD_V2_14B = Game14BInferencePipelineConfig(
    name=_SPEC["name"],
    encoder=Game14BCtrlEncoderConfig(
        encoder=WanVAEEncoderConfig(
            checkpoint_path=component_path(_SPEC, "vae"),
        ),
    ),
    decoder=WanVAEDecoderConfig(
        checkpoint_path=component_path(_SPEC, "vae"),
    ),
    image_encoder=CLIPImageEncoderConfig(
        model_id_or_local_path=component_path(_SPEC, "image_encoder"),
    ),
    text_encoder=UMT5TextEncoderConfig(
        model_id_or_local_path=component_path(_SPEC, "text_encoder"),
    ),
    # The first-frame condition is the cover's own latent plus a grey tail the
    # encoder caches per resolution, so nothing here pins its length: the
    # session passes ``total_blocks * len_t`` at cache build time, and a clip
    # shorter than the rollout makes a later AR step read past its end.
    diffusion_model=DiffusionModelConfig(
        seed=_SPEC["seed"],
        max_latent_frames=_SPEC["max_latent_frames"],
        # The AR cache commits the clean latent, so the cache-update forward
        # runs at t=0.
        context_noise=_SPEC["context_noise"],
        # One rank-0 draw over the whole chunk, broadcast to every shard.
        noise_from_rank0=_SPEC["noise_from_rank0"],
        transformer=Game14BTransformerConfig(
            # Cache-slot RoPE plus fp32 AdaLN, matching the training numerics.
            network=Game14BDiTNetworkConfig(
                apply_rope_before_kvcache=_SPEC["apply_rope_before_kvcache"],
                fp32_adaln=_SPEC["fp32_adaln"],
            ),
            dcp_checkpoint_path=GAME2V_EDEN_WORLD_V2_14B_CHECKPOINT_PATH,
            # Single-rollout layout: no leading batch dim.
            batch_shape=(),
            len_t=GAME2V_LEN_T,
            guidance_scale=_SPEC["guidance_scale"],
            window_size_t=GAME2V_WINDOW_SIZE_T,
            sink_size_t=_SPEC["sink_size_t"],
            # Stamp the cover's latent into the first chunk and hold it at t=0,
            # composed with the Wan 2.1 channel concat of mask + image latent.
            stamp_image_latent=_SPEC["stamp_image_latent"],
            ti2v_first_frame_per_token_timestep=_SPEC[
                "ti2v_first_frame_per_token_timestep"
            ],
            concat_image_mask_to_latent=_SPEC["concat_image_mask_to_latent"],
            compile_network=_SPEC["compile_network"],
        ),
        scheduler=FlowMatchSchedulerConfig(**scheduler_kwargs(_SPEC)),
    ),
)
"""Proxy-DINO conditioned AR 14B pipeline for the Game2V app."""


# Same model and schedule with the Wan VAE decoder swapped for the LightTAE
# tiny decoder. Compression ratios (temporal 4, spatial 8) and latent width
# (16) match the Wan VAE, so frame counts and the capture pacing are
# unchanged; only decode speed and fidelity differ.
PIPELINE_GAME2V_EDEN_WORLD_V2_14B_TAEHV = derive_config(
    PIPELINE_GAME2V_EDEN_WORLD_V2_14B,
    name=_SPEC["taehv"]["name"],
    decoder=TeahvVAEDecoderConfig(
        checkpoint_path=resolve_path(_SPEC, _SPEC["taehv"]["decoder"]),
    ),
)
"""The 14B pipeline with the LightTAE decoder instead of the Wan VAE."""


__all__ = [
    "GAME2V_EDEN_WORLD_V2_14B_CHECKPOINT_PATH",
    "GAME2V_LEN_T",
    "GAME2V_WINDOW_SIZE_T",
    "PIPELINE_GAME2V_EDEN_WORLD_V2_14B",
    "PIPELINE_GAME2V_EDEN_WORLD_V2_14B_TAEHV",
]
