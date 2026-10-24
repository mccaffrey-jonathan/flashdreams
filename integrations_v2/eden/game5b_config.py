# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config for the proxy-DINO conditioned AR Wan 2.2 TI2V-5B Game2V model.

The values live in :file:`configs/game2v_5b.yaml`, which this module loads and
maps onto the dataclasses:

* chunk size 3 -> :data:`GAME2V_5B_LEN_T`.
* 4 steps at timesteps ``[1000, 937.5, 833.3333, 625]`` (shift 5.0,
  stochastic sampling) -> the :class:`FlowMatchSchedulerConfig`.
* ``guidance_scale: 1.0`` -> CFG off, one conditional KV stream.
* An 18-frame attention budget with a 6-frame sink ->
  :data:`GAME2V_5B_WINDOW_SIZE_T` / :data:`GAME2V_5B_SINK_SIZE_T`.
* 704x1280 -> :data:`GAME2V_5B_PIXEL_HEIGHT` / ``_WIDTH``.
* The first frame packed into the first chunk -> the TI2V stamp knobs.

Kept out of :mod:`eden.config` because that module's ``PIPELINE_CONFIGS`` is
typed to the camera-control pipeline; this model conditions on rendered game
frames through DINOv3 instead of Plücker camera volumes, so it is a different
pipeline type entirely. It is also separate from :mod:`eden.game14b_config`:
that one is Wan 2.1 14B I2V (channel-concat first frame, CLIP branch, 16-channel
8x VAE), this one is Wan 2.2 TI2V 5B (stamped first frame, no CLIP, 48-channel
16x VAE).

Two consequences of the 5B's 16x-spatial VAE are worth stating, because both
differ from every other Game2V preset:

* **Resolution is pinned to 704x1280.** The DiT's per-frame token count is
  ``(H/16/2) * (W/16/2)``, and ``Wan21Transformer`` asserts that the KV window
  and sink token counts divide by the CP size. 704x1280 gives 22*40 = 880
  tokens per latent frame, which 8 ranks shard evenly; the 480x832 the 14B runs
  at would give 15*26 = 390 and fail that assert on an 8-GPU node.
* **The LightTAE variant needs its own weights.** The 14B's tiny decoder is
  8x-spatial over 16 latent channels and cannot read this VAE's 48-channel 16x
  latents, so :data:`PIPELINE_GAME2V_EDEN_WORLD_V2_5B_TAEHV` uses the
  ``lighttaew2_2`` checkpoint through
  :class:`~flashdreams.recipes.taehv.TeahvVAEWan22DecoderConfig` instead.
"""

from __future__ import annotations

from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler import FlowMatchSchedulerConfig
from flashdreams.infra.encoder.text.umt5 import UMT5TextEncoderConfig
from flashdreams.recipes.taehv import TeahvVAEWan22DecoderConfig
from flashdreams.recipes.wan.autoencoder.vae import (
    Wan22TI2V5BVAEDecoderConfig,
    Wan22TI2V5BVAEEncoderConfig,
)
from eden.impl.game5b.encoder import Game5BCtrlEncoderConfig
from eden.impl.game5b.network import Game5BDiTNetworkConfig
from eden.impl.game5b.pipeline import Game5BInferencePipelineConfig
from eden.impl.game5b.transformer import Game5BTransformerConfig
from eden.pipeline_yaml import (
    component_path,
    load_pipeline_yaml,
    resolve_path,
    scheduler_kwargs,
)

# Every number below comes from configs/game2v_5b.yaml; the provenance of each
# one is documented there. This module only maps them onto the dataclasses.
_SPEC = load_pipeline_yaml("game2v_5b")

GAME2V_EDEN_WORLD_V2_5B_CHECKPOINT_PATH = component_path(_SPEC, "transformer")
"""The distilled 5B DiT, published as a single safetensors file with Diffusers
keys. Read through
:func:`eden.impl.game5b.checkpoint.load_game5b_dit_state_dict`, which applies
the shared Diffusers remap.
"""

GAME2V_5B_LEN_T = _SPEC["len_t"]
GAME2V_5B_SINK_SIZE_T = _SPEC["sink_size_t"]
GAME2V_5B_WINDOW_SIZE_T = _SPEC["window_size_t"]
GAME2V_5B_PIXEL_HEIGHT = _SPEC["pixel_height"]
GAME2V_5B_PIXEL_WIDTH = _SPEC["pixel_width"]
GAME2V_5B_FPS = _SPEC["fps"]

PIPELINE_GAME2V_EDEN_WORLD_V2_5B = Game5BInferencePipelineConfig(
    name=_SPEC["name"],
    encoder=Game5BCtrlEncoderConfig(
        encoder=Wan22TI2V5BVAEEncoderConfig(
            checkpoint_path=component_path(_SPEC, "vae"),
        ),
    ),
    decoder=Wan22TI2V5BVAEDecoderConfig(
        checkpoint_path=component_path(_SPEC, "vae"),
    ),
    # Wan 2.2 TI2V-5B has no CLIP vision tower; the first frame is conditioned
    # through the clean-latent stamp instead.
    image_encoder=None,
    text_encoder=UMT5TextEncoderConfig(
        model_id_or_local_path=component_path(_SPEC, "text_encoder"),
    ),
    diffusion_model=DiffusionModelConfig(
        seed=_SPEC["seed"],
        max_latent_frames=_SPEC["max_latent_frames"],
        # The AR cache is committed from the clean latent, so the cache-update
        # forward runs at t=0.
        context_noise=_SPEC["context_noise"],
        # One rank-0 draw over the whole chunk, broadcast to every shard.
        noise_from_rank0=_SPEC["noise_from_rank0"],
        transformer=Game5BTransformerConfig(
            # Cache-slot RoPE: K cached unrotated, rotated by its arena slot.
            # fp32 AdaLN / LayerNorm / residuals, as the reference runs.
            network=Game5BDiTNetworkConfig(
                apply_rope_before_kvcache=_SPEC["apply_rope_before_kvcache"],
                fp32_adaln=_SPEC["fp32_adaln"],
            ),
            dcp_checkpoint_path=GAME2V_EDEN_WORLD_V2_5B_CHECKPOINT_PATH,
            # Single-rollout layout: no leading batch dim.
            batch_shape=(),
            len_t=GAME2V_5B_LEN_T,
            guidance_scale=_SPEC["guidance_scale"],
            window_size_t=GAME2V_5B_WINDOW_SIZE_T,
            sink_size_t=GAME2V_5B_SINK_SIZE_T,
            # The cover's latent rides inside the first denoising chunk rather
            # than being committed ahead of it. The stamp re-injects it before
            # every forward and the per-token timestep holds those tokens at t=0
            # while the rest of the chunk denoises. Wan 2.2 TI2V seeds the first
            # frame this way rather than by channel-concatenating a mask, so no
            # mask channels are appended.
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
"""Proxy-DINO conditioned AR 5B pipeline for the Game2V app."""


# Same model and schedule with the Wan 2.2 VAE decoder swapped for the
# ``lighttaew2_2`` tiny decoder. Its compression ratios (temporal 4, spatial 16)
# and latent width (48) match the Wan 2.2 TI2V-5B VAE, so frame counts, latent
# sizes and the renderer's capture pacing are all unchanged; only decode speed
# and fidelity differ. The *encoder* stays the real VAE: the first frame is
# encoded once, so there is nothing to win there, and the stamped condition
# latent has to live in the same space the DiT was trained on.
PIPELINE_GAME2V_EDEN_WORLD_V2_5B_TAEHV = derive_config(
    PIPELINE_GAME2V_EDEN_WORLD_V2_5B,
    name=_SPEC["taehv"]["name"],
    decoder=TeahvVAEWan22DecoderConfig(
        checkpoint_path=resolve_path(_SPEC, _SPEC["taehv"]["decoder"]),
    ),
)
"""The 5B pipeline with the LightTAE Wan 2.2 decoder instead of the Wan VAE."""


__all__ = [
    "GAME2V_5B_FPS",
    "GAME2V_5B_LEN_T",
    "GAME2V_5B_PIXEL_HEIGHT",
    "GAME2V_5B_PIXEL_WIDTH",
    "GAME2V_5B_SINK_SIZE_T",
    "GAME2V_5B_WINDOW_SIZE_T",
    "GAME2V_EDEN_WORLD_V2_5B_CHECKPOINT_PATH",
    "PIPELINE_GAME2V_EDEN_WORLD_V2_5B",
    "PIPELINE_GAME2V_EDEN_WORLD_V2_5B_TAEHV",
]
