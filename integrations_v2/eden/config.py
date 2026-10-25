# SPDX-FileCopyrightText: Copyright (c) 2026 Vast AI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for FlashDreams v0.2.0: merged from eden/game5b_config.py and
# eden/game14b_config.py (VAST EdenLive realtime integration).

"""Pipeline configs for the EdenLive proxy-DINO conditioned AR models.

EdenLive renders a *reference* (proxy) video -- a greybox game, a capture, any
rough render -- into detailed frames, guided by a target first frame and a text
prompt. The proxy reaches the DiT through a post-trained DINOv3 encoder whose
1024 projected channels are concatenated onto the latent channels.

Two models are published (``VAST-AI/EdenLive`` on Hugging Face):

* **5B** -- Wan 2.2 TI2V-5B, 704x1280 at 24 fps, chunk 3, 4 steps at timesteps
  ``[1000, 937.5, 833.33, 625]`` (shift 5), an 18-frame attention budget split
  as 6 sink + 12 rolling, first frame stamped into chunk 0 at ``t=0``. This is
  the model that fits one 32 GB GeForce card.
* **14B** -- Wan 2.1 I2V-14B, 480x832 at 16 fps, shift 3, CLIP branch, the
  first frame channel-concatenated. Its bf16 weights plus KV cache do not fit
  a 32 GB card at the published window.

The numbers live in :file:`configs/game2v_5b.yaml` and
:file:`configs/game2v_14b.yaml`; this module maps them onto the dataclasses.
Weight paths resolve against ``model_root`` in the YAML, else the
``EDEN_MODEL_ROOT`` environment variable (see :mod:`eden.impl.model_root`).

The 5B's 16x-spatial VAE pins the resolution: the DiT's per-frame token count
is ``(H/16/2) * (W/16/2)``, and 704x1280 gives 22*40 = 880 tokens per latent
frame. The LightTAE variant needs the 48-channel ``lighttaew2_2`` decoder
(:class:`~eden.impl.taehv_wan22.TaehvVAEWan22DecoderConfig`); the 14B's tiny
decoder is 8x over 16 channels and cannot read these latents.
"""

from __future__ import annotations

from typing import Any

from eden.impl.diffusion import EdenDiffusionModelConfig
from eden.impl.game5b.encoder import Game5BCtrlEncoderConfig
from eden.impl.game5b.network import Game5BDiTNetworkConfig
from eden.impl.game5b.pipeline import Game5BInferencePipelineConfig
from eden.impl.game5b.transformer import Game5BTransformerConfig
from eden.impl.game14b.encoder import Game14BCtrlEncoderConfig
from eden.impl.game14b.network import Game14BDiTNetworkConfig
from eden.impl.game14b.pipeline import Game14BInferencePipelineConfig
from eden.impl.game14b.transformer import Game14BTransformerConfig
from eden.impl.pipeline_yaml import (
    component_path,
    load_pipeline_yaml,
    resolve_path,
    scheduler_kwargs,
)
from eden.impl.scheduler import EdenFlowMatchSchedulerConfig
from eden.impl.taehv_wan22 import TaehvVAEWan22DecoderConfig
from eden.impl.text_encoder import EdenUMT5TextEncoderConfig
from eden.impl.vae import EdenWan22TI2V5BVAEEncoderConfig
from flashdreams.infra.config import derive_config
from flashdreams.infra.encoder.image.clip import CLIPImageEncoderConfig
from flashdreams.recipes.taehv import TaehvVAEDecoderConfig
from flashdreams.recipes.wan.autoencoder.vae import (
    Wan22TI2V5BVAEDecoderConfig,
    WanVAEDecoderConfig,
    WanVAEEncoderConfig,
)

# ---------------------------------------------------------------------------
# 5B: Wan 2.2 TI2V-5B, proxy-DINO conditioned, AR
# ---------------------------------------------------------------------------

GAME2V_5B_SPEC: dict[str, Any] = load_pipeline_yaml("game2v_5b")
"""The parsed ``configs/game2v_5b.yaml``; every 5B number below comes from it."""

GAME2V_EDEN_WORLD_V2_5B_CHECKPOINT_PATH = component_path(GAME2V_5B_SPEC, "transformer")
"""The distilled 5B DiT: one fp32 safetensors file with Diffusers keys, read by
:func:`eden.impl.game5b.checkpoint.load_game5b_dit_state_dict`."""

GAME2V_5B_LEN_T: int = GAME2V_5B_SPEC["len_t"]
GAME2V_5B_SINK_SIZE_T: int = GAME2V_5B_SPEC["sink_size_t"]
GAME2V_5B_WINDOW_SIZE_T: int = GAME2V_5B_SPEC["window_size_t"]
GAME2V_5B_PIXEL_HEIGHT: int = GAME2V_5B_SPEC["pixel_height"]
GAME2V_5B_PIXEL_WIDTH: int = GAME2V_5B_SPEC["pixel_width"]
GAME2V_5B_FPS: int = GAME2V_5B_SPEC["fps"]

PIPELINE_GAME2V_EDEN_WORLD_V2_5B = Game5BInferencePipelineConfig(
    name=GAME2V_5B_SPEC["name"],
    encoder=Game5BCtrlEncoderConfig(
        encoder=EdenWan22TI2V5BVAEEncoderConfig(
            checkpoint_path=component_path(GAME2V_5B_SPEC, "vae"),
        ),
    ),
    decoder=Wan22TI2V5BVAEDecoderConfig(
        checkpoint_path=component_path(GAME2V_5B_SPEC, "vae"),
    ),
    # Wan 2.2 TI2V-5B has no CLIP vision tower; the first frame is conditioned
    # through the clean-latent stamp instead.
    image_encoder=None,
    text_encoder=EdenUMT5TextEncoderConfig(
        model_id_or_local_path=component_path(GAME2V_5B_SPEC, "text_encoder"),
    ),
    diffusion_model=EdenDiffusionModelConfig(
        seed=GAME2V_5B_SPEC["seed"],
        max_latent_frames=GAME2V_5B_SPEC["max_latent_frames"],
        # The AR cache commits the clean latent, so the cache-update forward
        # runs at t=0.
        context_noise=GAME2V_5B_SPEC["context_noise"],
        # Noise drawn in the unpatchified (C, T, H, W) layout, as the reference.
        noise_from_rank0=GAME2V_5B_SPEC["noise_from_rank0"],
        transformer=Game5BTransformerConfig(
            # Cache-slot RoPE: K cached unrotated, rotated by its arena slot.
            # fp32 AdaLN / LayerNorm / residuals, as the reference runs.
            network=Game5BDiTNetworkConfig(
                apply_rope_before_kvcache=GAME2V_5B_SPEC["apply_rope_before_kvcache"],
                fp32_adaln=GAME2V_5B_SPEC["fp32_adaln"],
            ),
            dcp_checkpoint_path=GAME2V_EDEN_WORLD_V2_5B_CHECKPOINT_PATH,
            # Single-rollout layout: no leading batch dim.
            batch_shape=(),
            len_t=GAME2V_5B_LEN_T,
            guidance_scale=GAME2V_5B_SPEC["guidance_scale"],
            window_size_t=GAME2V_5B_WINDOW_SIZE_T,
            sink_size_t=GAME2V_5B_SINK_SIZE_T,
            # The cover's latent rides inside the first denoising chunk: the
            # stamp re-injects it before every forward and the per-token
            # timestep holds those tokens at t=0. No mask channels.
            stamp_image_latent=GAME2V_5B_SPEC["stamp_image_latent"],
            ti2v_first_frame_per_token_timestep=GAME2V_5B_SPEC[
                "ti2v_first_frame_per_token_timestep"
            ],
            concat_image_mask_to_latent=GAME2V_5B_SPEC["concat_image_mask_to_latent"],
            compile_network=GAME2V_5B_SPEC["compile_network"],
        ),
        scheduler=EdenFlowMatchSchedulerConfig(**scheduler_kwargs(GAME2V_5B_SPEC)),
    ),
)
"""Proxy-DINO conditioned AR 5B pipeline with the full Wan 2.2 VAE decoder."""

# The *encoder* stays the real VAE: the first frame is encoded once, and the
# stamped latent has to live in the space the DiT was trained on. Only decode
# speed and fidelity change; frame counts and latent sizes do not.
PIPELINE_GAME2V_EDEN_WORLD_V2_5B_TAEHV = derive_config(
    PIPELINE_GAME2V_EDEN_WORLD_V2_5B,
    name=GAME2V_5B_SPEC["taehv"]["name"],
    decoder=TaehvVAEWan22DecoderConfig(
        checkpoint_path=resolve_path(
            GAME2V_5B_SPEC, GAME2V_5B_SPEC["taehv"]["decoder"]
        ),
    ),
)
"""The 5B pipeline with the LightTAE Wan 2.2 decoder instead of the Wan VAE."""

# Single 32 GB GeForce (RTX 5090) preset: dynamic row-wise FP8 block linears
# plus the in-tree Sage-style FP8 self-attention, with torch.compile and CUDA
# graphs. Measured on an RTX 5090 (Windows, torch 2.12.1+cu130): ~18 GiB peak,
# ~0.64 s per 12-frame chunk once the KV window is full, with output inside the
# bf16 run-to-run noise floor. The first run compiles one graph per KV-fill
# shape (several minutes) before reaching steady state.
PIPELINE_GAME2V_EDEN_WORLD_V2_5B_TAEHV_RTX5090 = derive_config(
    PIPELINE_GAME2V_EDEN_WORLD_V2_5B_TAEHV,
    name=f"{GAME2V_5B_SPEC['taehv']['name']}-rtx-5090",
    diffusion_model=dict(
        transformer=dict(
            network=derive_config(
                PIPELINE_GAME2V_EDEN_WORLD_V2_5B_TAEHV.diffusion_model.transformer.network,
                linear_backend="rowwise_fp8",
                self_attention_backend="sage_fp8",
            ),
            compile_network=True,
            use_cuda_graph=True,
        ),
    ),
)
"""The 5B LightTAE pipeline with FP8 linears and FP8 Sage attention (single GPU)."""

# ---------------------------------------------------------------------------
# 14B: Wan 2.1 I2V-14B, proxy-DINO conditioned, AR
# ---------------------------------------------------------------------------

GAME2V_14B_SPEC: dict[str, Any] = load_pipeline_yaml("game2v_14b")
"""The parsed ``configs/game2v_14b.yaml``."""

GAME2V_EDEN_WORLD_V2_14B_CHECKPOINT_PATH = component_path(
    GAME2V_14B_SPEC, "transformer"
)
"""The distilled 14B DiT, read by
:func:`eden.impl.game14b.checkpoint.load_game14b_dit_state_dict`."""

GAME2V_14B_LEN_T: int = GAME2V_14B_SPEC["len_t"]
GAME2V_14B_WINDOW_SIZE_T: int = GAME2V_14B_SPEC["window_size_t"]
GAME2V_14B_PIXEL_HEIGHT: int = GAME2V_14B_SPEC["pixel_height"]
GAME2V_14B_PIXEL_WIDTH: int = GAME2V_14B_SPEC["pixel_width"]
GAME2V_14B_FPS: int = GAME2V_14B_SPEC["fps"]

PIPELINE_GAME2V_EDEN_WORLD_V2_14B = Game14BInferencePipelineConfig(
    name=GAME2V_14B_SPEC["name"],
    encoder=Game14BCtrlEncoderConfig(
        encoder=WanVAEEncoderConfig(
            checkpoint_path=component_path(GAME2V_14B_SPEC, "vae"),
        ),
    ),
    decoder=WanVAEDecoderConfig(
        checkpoint_path=component_path(GAME2V_14B_SPEC, "vae"),
    ),
    image_encoder=CLIPImageEncoderConfig(
        model_id_or_local_path=component_path(GAME2V_14B_SPEC, "image_encoder"),  # type: ignore[arg-type]
    ),
    text_encoder=EdenUMT5TextEncoderConfig(
        model_id_or_local_path=component_path(GAME2V_14B_SPEC, "text_encoder"),
    ),
    # The first-frame condition is the cover's own latent plus a grey tail the
    # encoder caches per resolution: the session passes ``total_blocks * len_t``
    # at cache build time.
    diffusion_model=EdenDiffusionModelConfig(
        seed=GAME2V_14B_SPEC["seed"],
        max_latent_frames=GAME2V_14B_SPEC["max_latent_frames"],
        context_noise=GAME2V_14B_SPEC["context_noise"],
        noise_from_rank0=GAME2V_14B_SPEC["noise_from_rank0"],
        transformer=Game14BTransformerConfig(
            network=Game14BDiTNetworkConfig(
                apply_rope_before_kvcache=GAME2V_14B_SPEC["apply_rope_before_kvcache"],
                fp32_adaln=GAME2V_14B_SPEC["fp32_adaln"],
            ),
            dcp_checkpoint_path=GAME2V_EDEN_WORLD_V2_14B_CHECKPOINT_PATH,
            batch_shape=(),
            len_t=GAME2V_14B_LEN_T,
            guidance_scale=GAME2V_14B_SPEC["guidance_scale"],
            window_size_t=GAME2V_14B_WINDOW_SIZE_T,
            sink_size_t=GAME2V_14B_SPEC["sink_size_t"],
            # Stamp the cover's latent into the first chunk and hold it at t=0,
            # composed with the Wan 2.1 channel concat of mask + image latent.
            stamp_image_latent=GAME2V_14B_SPEC["stamp_image_latent"],
            ti2v_first_frame_per_token_timestep=GAME2V_14B_SPEC[
                "ti2v_first_frame_per_token_timestep"
            ],
            concat_image_mask_to_latent=GAME2V_14B_SPEC["concat_image_mask_to_latent"],
            compile_network=GAME2V_14B_SPEC["compile_network"],
        ),
        scheduler=EdenFlowMatchSchedulerConfig(**scheduler_kwargs(GAME2V_14B_SPEC)),
    ),
)
"""Proxy-DINO conditioned AR 14B pipeline with the full Wan 2.1 VAE decoder."""

PIPELINE_GAME2V_EDEN_WORLD_V2_14B_TAEHV = derive_config(
    PIPELINE_GAME2V_EDEN_WORLD_V2_14B,
    name=GAME2V_14B_SPEC["taehv"]["name"],
    decoder=TaehvVAEDecoderConfig(
        checkpoint_path=resolve_path(
            GAME2V_14B_SPEC, GAME2V_14B_SPEC["taehv"]["decoder"]
        ),
    ),
)
"""The 14B pipeline with the LightTAE decoder instead of the Wan VAE."""


__all__ = [
    "GAME2V_14B_FPS",
    "GAME2V_14B_LEN_T",
    "GAME2V_14B_PIXEL_HEIGHT",
    "GAME2V_14B_PIXEL_WIDTH",
    "GAME2V_14B_SPEC",
    "GAME2V_14B_WINDOW_SIZE_T",
    "GAME2V_5B_FPS",
    "GAME2V_5B_LEN_T",
    "GAME2V_5B_PIXEL_HEIGHT",
    "GAME2V_5B_PIXEL_WIDTH",
    "GAME2V_5B_SINK_SIZE_T",
    "GAME2V_5B_SPEC",
    "GAME2V_5B_WINDOW_SIZE_T",
    "GAME2V_EDEN_WORLD_V2_14B_CHECKPOINT_PATH",
    "GAME2V_EDEN_WORLD_V2_5B_CHECKPOINT_PATH",
    "PIPELINE_GAME2V_EDEN_WORLD_V2_14B",
    "PIPELINE_GAME2V_EDEN_WORLD_V2_14B_TAEHV",
    "PIPELINE_GAME2V_EDEN_WORLD_V2_5B",
    "PIPELINE_GAME2V_EDEN_WORLD_V2_5B_TAEHV",
    "PIPELINE_GAME2V_EDEN_WORLD_V2_5B_TAEHV_RTX5090",
]
