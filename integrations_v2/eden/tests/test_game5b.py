# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the proxy-DINO conditioned AR 5B Game2V preset.

Every assertion here traces to the training recipe or to the checkpoint's own
tensor shapes, so a drift in either is what these are meant to catch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from eden.config import (
    GAME2V_5B_FPS,
    GAME2V_5B_LEN_T,
    GAME2V_5B_PIXEL_HEIGHT,
    GAME2V_5B_PIXEL_WIDTH,
    GAME2V_5B_SINK_SIZE_T,
    GAME2V_5B_WINDOW_SIZE_T,
    PIPELINE_GAME2V_EDEN_WORLD_V2_5B,
    PIPELINE_GAME2V_EDEN_WORLD_V2_5B_TAEHV,
)
from eden.impl.game5b.network import (
    GAME5B_CONTROL_DIM,
    GAME5B_LATENT_CHANNELS,
    Game5BDiTNetworkConfig,
)

pytestmark = pytest.mark.ci_cpu


def test_channel_layout_matches_the_checkpoint_patch_embedding() -> None:
    """1072 input channels: 48 noisy latent + 1024 proxy-DINO control.

    The checkpoint's ``patch_embedding.weight`` is ``[3072, 1072, 1, 2, 2]``.
    Unlike the 14B AR variant there are no mask / condition-latent channels:
    Wan 2.2 TI2V seeds the first frame by stamping a clean latent, not by
    channel-concatenating a mask.
    """
    network = PIPELINE_GAME2V_EDEN_WORLD_V2_5B.diffusion_model.transformer.network
    assert GAME5B_LATENT_CHANNELS == 48
    assert GAME5B_CONTROL_DIM == 1024
    assert network.in_dim == GAME5B_LATENT_CHANNELS + GAME5B_CONTROL_DIM == 1072
    assert network.control_dim == GAME5B_CONTROL_DIM
    # The 5B head still emits the 48 latent channels the VAE decodes.
    assert network.out_dim == GAME5B_LATENT_CHANNELS
    # Wan 2.2 TI2V-5B has no CLIP vision tower, so no image cross-attention.
    assert network.cross_attn_enable_img is False
    assert PIPELINE_GAME2V_EDEN_WORLD_V2_5B.image_encoder is None
    # 5B shape: 30 layers of 3072 wide, 24 heads.
    assert (network.dim, network.num_layers, network.num_heads) == (3072, 30, 24)


def test_control_dim_must_agree_with_in_dim() -> None:
    """A mismatched control width is rejected rather than silently loaded."""
    with pytest.raises(ValueError, match="channel-concat layout"):
        Game5BDiTNetworkConfig(in_dim=1060)


def test_packed_first_frame_uses_the_ti2v_stamp() -> None:
    """``pack_first_frame: true`` maps onto the TI2V stamp, not a concat.

    The cover latent rides inside the first denoising chunk: re-stamped before
    every forward and held at ``t=0`` by the per-token timestep, while the rest
    of the chunk denoises at the scheduler step.
    """
    transformer = PIPELINE_GAME2V_EDEN_WORLD_V2_5B.diffusion_model.transformer
    assert transformer.stamp_image_latent is True
    assert transformer.ti2v_first_frame_per_token_timestep is True
    assert transformer.first_frame_timestep_value == 0.0
    # No mask channels are appended; that is the 14B's channel-concat path.
    assert transformer.concat_image_mask_to_latent is False


def test_sampling_matches_the_reference_schedule() -> None:
    """CFG off, 4 stochastic steps, and the reference's own timestep grid."""
    pipeline = PIPELINE_GAME2V_EDEN_WORLD_V2_5B
    transformer = pipeline.diffusion_model.transformer
    scheduler = pipeline.diffusion_model.scheduler
    # ``guidance_scale: 1.0`` upstream: one conditional KV stream.
    assert transformer.guidance_scale == 1.0
    # ``chunk_size: 3``.
    assert transformer.len_t == GAME2V_5B_LEN_T == 3
    # ``num_inference_steps: 4`` at ``shift: 5.0``.
    assert scheduler.num_inference_steps == 4
    assert scheduler.shift == 5.0
    assert scheduler.denoising_timesteps == [1000, 750, 500, 250]
    assert scheduler.warp_denoising_step is True
    # ``context_noise_sigma: 0.0``: the AR cache commits the clean latent.
    assert pipeline.diffusion_model.context_noise == 0
    # One rank-0 noise draw over the whole chunk, broadcast to every shard.
    assert pipeline.diffusion_model.noise_from_rank0 is True
    # Cache-slot RoPE: K cached unrotated, rotated by its arena slot.
    assert transformer.network.apply_rope_before_kvcache is False
    # Timesteps reach the network in fp32 so bf16 does not round them.
    assert scheduler.network_timestep_dtype == torch.float32


def test_denoising_timesteps_reproduce_the_reference_timestep_list() -> None:
    """The warped grid must equal upstream's explicit ``scheduler.timesteps``.

    The reference pins ``[1000, 937.5, 833.3333333333, 625]``. FlashDreams
    instead indexes the warped sigma table at ``1000 - t``, so this check is what
    ties the two representations together.
    """
    scheduler = PIPELINE_GAME2V_EDEN_WORLD_V2_5B.diffusion_model.scheduler
    n = scheduler.num_train_timesteps
    sigmas = torch.linspace(
        scheduler.sigma_max, scheduler.sigma_min, n + 1, dtype=torch.float32
    )[:-1]
    shift = scheduler.shift
    warped = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    got = [float(warped[n - t] * n) for t in scheduler.denoising_timesteps]
    expected = [1000.0, 937.5, 833.3333333333, 625.0]
    assert got == pytest.approx(expected, abs=1e-4)


def test_kv_budget_matches_the_reference_context_window() -> None:
    """18 total latent frames of context, 6 of them a fixed sink.

    Upstream's ``max_latent_frames: 18`` is the whole attention budget with
    ``sink_latent_frames: 6`` carved out of it; FlashDreams sizes the arena as
    ``sink + window``, so the same budget is 6 + 12.
    """
    transformer = PIPELINE_GAME2V_EDEN_WORLD_V2_5B.diffusion_model.transformer
    assert transformer.sink_size_t == GAME2V_5B_SINK_SIZE_T == 6
    assert transformer.window_size_t == GAME2V_5B_WINDOW_SIZE_T == 12
    assert transformer.sink_size_t + transformer.window_size_t == 18
    # The KV arena must hold a whole number of AR chunks, which is what the
    # CUDA-graph capture threshold is derived from.
    assert (transformer.sink_size_t + transformer.window_size_t) % GAME2V_5B_LEN_T == 0


def test_frame_size_shards_across_eight_ranks() -> None:
    """704x1280 is required, not preferred: 480x832 fails the CP assert.

    The DiT token count per latent frame is ``(H/16/2) * (W/16/2)`` under the
    5B's 16x VAE. ``Wan21Transformer`` asserts the window and sink token counts
    divide by the CP size, so a frame size whose token count does not shard
    across 8 ranks cannot run on an 8-GPU node.
    """
    vae_ratio = 16
    patch = 2
    cp_size = 8
    transformer = PIPELINE_GAME2V_EDEN_WORLD_V2_5B.diffusion_model.transformer

    def tokens_per_frame(height: int, width: int) -> int:
        return (height // vae_ratio // patch) * (width // vae_ratio // patch)

    pinned = tokens_per_frame(GAME2V_5B_PIXEL_HEIGHT, GAME2V_5B_PIXEL_WIDTH)
    assert (GAME2V_5B_PIXEL_HEIGHT, GAME2V_5B_PIXEL_WIDTH) == (704, 1280)
    # 704/16 x 1280/16 = 44x80 latent, patchified 2x2 -> 22x40 = 880 tokens.
    assert pinned == 22 * 40 == 880
    for frames in (
        transformer.len_t,
        transformer.window_size_t,
        transformer.sink_size_t,
    ):
        assert (frames * pinned) % cp_size == 0

    # The 14B's frame size would not shard: 390 tokens/frame, and 3 * 390 = 1170.
    assert (transformer.len_t * tokens_per_frame(480, 832)) % cp_size != 0


def test_vae_is_the_wan22_ti2v_5b_autoencoder() -> None:
    """Encoder and decoder must both be the 48-channel 16x 5B VAE.

    A latent encoded by one VAE and decoded by another is silently wrong rather
    than an error, so the two are pinned together here.
    """
    pipeline = PIPELINE_GAME2V_EDEN_WORLD_V2_5B
    encoder = pipeline.encoder.encoder
    decoder = pipeline.decoder
    assert encoder.z_dim == decoder.z_dim == GAME5B_LATENT_CHANNELS
    assert encoder.patch_size == decoder.patch_size == 2
    assert encoder.is_residual is decoder.is_residual is True
    assert encoder.checkpoint_path == decoder.checkpoint_path
    # 8x encoder body * 2x patchify = the 16x ratio the control stream assumes.
    from eden.apps.game2v.game5b import GAME2V_5B_VAE_SPATIAL_RATIO

    assert GAME2V_5B_VAE_SPATIAL_RATIO == 16


def test_taehv_variant_swaps_only_the_decoder() -> None:
    """The Wan 2.2 LightTAE matches the 5B VAE's ratios, so pacing is unchanged.

    The tiny decoder's own class constant is the 8x Hunyuan body ratio; Wan 2.2's
    branch adds a 2x2 pixel-shuffle on top of it. If the config's ratio did not
    end up at 16 the latent grid would be sized from ``704 / 8 = 88`` instead of
    44, and the whole patch grid would misalign with no error attached.
    """
    from eden.impl.taehv_wan22 import (
        TaehvVAEWan22Decoder,
        TaehvVAEWan22DecoderConfig,
    )

    from flashdreams.recipes.wan.autoencoder.vae import WanVAE

    base = PIPELINE_GAME2V_EDEN_WORLD_V2_5B
    variant = PIPELINE_GAME2V_EDEN_WORLD_V2_5B_TAEHV
    assert isinstance(variant.decoder, TaehvVAEWan22DecoderConfig)
    assert variant.name == "game2v-eden-world-v2-5b-taehv"
    # Only the decoder moves: the encoder stays the real VAE, because the DiT's
    # stamped condition latent has to live in the space it was trained on.
    assert variant.encoder == base.encoder
    assert variant.diffusion_model == base.diffusion_model
    assert TaehvVAEWan22Decoder.TEMPORAL_COMPRESSION_RATIO == 4
    assert TaehvVAEWan22Decoder.SPATIAL_COMPRESSION_RATIO == 16
    # The Wan 2.2 5B VAE it stands in for: 8x encoder body * 2x patchify.
    assert WanVAE.SPATIAL_COMPRESSION_RATIO * base.decoder.patch_size == 16
    assert WanVAE.TEMPORAL_COMPRESSION_RATIO == 4


def test_taehv_variant_reads_the_wan22_tiny_weights() -> None:
    """``lighttaew2_2`` is the only checkpoint with 48 latent channels.

    ``lighttaew2_1`` would load a 16-channel first conv against 48-channel
    latents, so the checkpoint and the ``wan22`` architecture branch have to be
    selected together.
    """
    from eden.config import GAME2V_5B_SPEC
    from eden.impl.pipeline_yaml import resolve_path
    from eden.impl.taehv_wan22 import TaehvVAEWan22DecoderConfig

    config = PIPELINE_GAME2V_EDEN_WORLD_V2_5B_TAEHV.decoder
    assert isinstance(config, TaehvVAEWan22DecoderConfig)
    # The variant reads the path the pipeline YAML names.
    assert config.checkpoint_path == resolve_path(
        GAME2V_5B_SPEC, GAME2V_5B_SPEC["taehv"]["decoder"]
    )
    assert config.checkpoint_path.endswith("decoder-5b/lighttaew2_2.pth")


def test_taehv_variant_denormalizes_with_the_wan22_vae_statistics() -> None:
    """The tiny decoder reads raw latents; the pipeline's are normalized.

    ``TaehvVAEDecoder.forward`` undoes the pipeline's normalization with these
    constants, so they have to be the same 48 values the Wan 2.2 VAE decoder
    would use. The 14B's 16-channel table is both the wrong length and the wrong
    distribution.
    """
    from eden.impl.taehv_wan22 import TaehvVAEWan22Decoder

    from flashdreams.recipes.taehv import TaehvVAEDecoder
    from flashdreams.recipes.wan.autoencoder.vae import (
        _WAN22_TI2V_5B_LATENT_MEAN,
        _WAN22_TI2V_5B_LATENT_STD,
    )

    assert TaehvVAEWan22Decoder._LIGHTTAE_MEAN == tuple(_WAN22_TI2V_5B_LATENT_MEAN)
    assert TaehvVAEWan22Decoder._LIGHTTAE_STD == tuple(_WAN22_TI2V_5B_LATENT_STD)
    assert len(TaehvVAEWan22Decoder._LIGHTTAE_MEAN) == GAME5B_LATENT_CHANNELS
    assert len(TaehvVAEDecoder._LIGHTTAE_MEAN) == 16


def test_taehv_app_defaults_keep_the_5b_wiring() -> None:
    """The TAE shell differs from the 5B's only in its decoder."""
    import dataclasses

    from eden.apps.game2v.adapter import (
        GAME2V_EDEN_WORLD_V2_5B_DEFAULTS,
    )
    from eden.apps.game2v.adapter import (
        GAME2V_EDEN_WORLD_V2_5B_TAEHV_DEFAULTS as defaults,
    )

    assert defaults.pipeline_config.name == "game2v-eden-world-v2-5b-taehv"
    assert defaults.pixel_height == GAME2V_5B_PIXEL_HEIGHT
    assert defaults.pixel_width == GAME2V_5B_PIXEL_WIDTH
    assert defaults.fps == GAME2V_5B_FPS
    assert defaults.control_stream_factory is (
        GAME2V_EDEN_WORLD_V2_5B_DEFAULTS.control_stream_factory
    )
    moved = {
        field.name
        for field in dataclasses.fields(defaults)
        if getattr(defaults, field.name)
        != getattr(GAME2V_EDEN_WORLD_V2_5B_DEFAULTS, field.name)
    }
    assert moved == {"pipeline_config"}


def test_fps_matches_the_reference() -> None:
    """``fps: 24`` upstream; the renderer paces its capture against it."""
    from eden.apps.game2v.adapter import GAME2V_EDEN_WORLD_V2_5B_DEFAULTS

    assert GAME2V_5B_FPS == 24
    assert GAME2V_EDEN_WORLD_V2_5B_DEFAULTS.fps == GAME2V_5B_FPS
    assert GAME2V_EDEN_WORLD_V2_5B_DEFAULTS.pixel_height == GAME2V_5B_PIXEL_HEIGHT
    assert GAME2V_EDEN_WORLD_V2_5B_DEFAULTS.pixel_width == GAME2V_5B_PIXEL_WIDTH


def test_app_defaults_wire_the_5b_hooks() -> None:
    """The app must use the 5B pipeline, resolver, step and control stream."""
    from eden.apps.game2v.adapter import GAME2V_EDEN_WORLD_V2_5B_DEFAULTS as defaults

    assert defaults.pipeline_config.name == "game2v-eden-world-v2-5b"
    # Per-chunk prompts are gated on this hook being set, so a missing control
    # stream factory would silently reject ``--dynamic-prompt``.
    assert defaults.control_stream_factory is not None
    assert defaults.generate_step is not None
    # The cover frame comes from the game registry, so no example data.
    assert defaults.input_defaults["example_data"] is False


def test_checkpoint_points_at_the_published_ema_file() -> None:
    """The published file is the distilled EMA, so there is no subtree to pick.

    The live generator, the DMD critic and both optimizer states are not in the
    weight tree at all, which is what makes sampling the wrong one impossible.
    """
    from eden.impl.model_root import EDEN_MODEL_ROOT

    transformer = PIPELINE_GAME2V_EDEN_WORLD_V2_5B.diffusion_model.transformer
    assert transformer.dcp_checkpoint_path == (
        f"{EDEN_MODEL_ROOT}/dit-5b/diffusion_pytorch_model.safetensors"
    )
    # The remap loader reads this slot; the base safetensors slot stays unset.
    assert transformer.checkpoint_path is None


def test_checkpoint_keys_need_only_the_shared_diffusers_remap() -> None:
    """The 5B needs no extra key remap, unlike the 14B.

    No CLIP cross-attention branch and no ``condition_norm``, so the shared
    table covers every key. Verified against the checkpoint metadata: 825 keys,
    none missing, none extra.
    """
    from eden.impl.game5b.checkpoint import game5b_dit_state_dict_from_diffusers

    mapped = game5b_dit_state_dict_from_diffusers(
        {
            "patch_embedding.weight": torch.zeros(1),
            "condition_embedder.text_embedder.linear_1.weight": torch.zeros(1),
            "blocks.0.attn1.to_q.weight": torch.zeros(1),
            "blocks.0.attn2.to_k.weight": torch.zeros(1),
            "blocks.0.ffn.net.0.proj.weight": torch.zeros(1),
            "scale_shift_table": torch.zeros(1),
            "proj_out.weight": torch.zeros(1),
        }
    )
    assert set(mapped) == {
        # The widened conv passes through unrenamed.
        "patch_embedding.weight",
        "text_embedding.0.weight",
        "blocks.0.self_attn.q.weight",
        "blocks.0.cross_attn.k.weight",
        "blocks.0.ffn.0.weight",
        "head.modulation",
        "head.head.weight",
    }


def test_encoder_stamps_only_the_first_chunk() -> None:
    """AR 0 encodes the cover; later steps emit a zero latent and zero mask.

    Past AR 0 the mask is all zeros, so the stamp is the identity and the
    condition latent never reaches the network. Skipping the VAE call there is
    bit-identical and saves one encode per chunk.
    """
    from eden.impl.game5b.encoder import (
        Game5BCtrlEncoder,
        Game5BCtrlEncoderCache,
        Game5BCtrlInput,
    )

    len_t = GAME2V_5B_LEN_T
    encodes = {"count": 0}

    class _VAEEncoder:
        temporal_compression_ratio = 4
        spatial_compression_ratio = 16

        def __call__(self, pixels, autoregressive_index, cache):
            del autoregressive_index, cache
            encodes["count"] += 1
            assert pixels.shape[-4] == 1
            return torch.arange(
                GAME5B_LATENT_CHANNELS * 2 * 2, dtype=torch.float32
            ).reshape(1, GAME5B_LATENT_CHANNELS, 2, 2)

    encoder = Game5BCtrlEncoder.__new__(Game5BCtrlEncoder)
    encoder.encoder = _VAEEncoder()
    cache = Game5BCtrlEncoderCache(vae={})
    control = torch.zeros(len_t, GAME5B_CONTROL_DIM, 2, 2)

    first = encoder.forward(
        Game5BCtrlInput(image=torch.zeros(1, 3, 32, 32), control=control),
        autoregressive_index=0,
        cache=cache,
    )
    assert encodes["count"] == 1
    # One-hot on latent frame 0: the packed condition frame, and nothing else.
    assert first.i2v.mask[0].eq(1.0).all()
    assert first.i2v.mask[1:].eq(0.0).all()
    assert first.i2v.latent.shape == (len_t, GAME5B_LATENT_CHANNELS, 2, 2)
    assert first.control is not None

    second = encoder.forward(
        Game5BCtrlInput(image=None, control=control),
        autoregressive_index=1,
        cache=cache,
    )
    # No second VAE call, and nothing for the stamp to inject.
    assert encodes["count"] == 1
    assert second.i2v.mask.eq(0.0).all()
    assert second.i2v.latent.eq(0.0).all()
    assert second.i2v.latent.shape == first.i2v.latent.shape


def test_encoder_requires_the_cover_at_ar_zero() -> None:
    """The packed first frame is encoded at AR 0 and nowhere else."""
    from eden.impl.game5b.encoder import (
        Game5BCtrlEncoder,
        Game5BCtrlEncoderCache,
        Game5BCtrlInput,
    )

    encoder = Game5BCtrlEncoder.__new__(Game5BCtrlEncoder)
    encoder.encoder = SimpleNamespace()
    control = torch.zeros(GAME2V_5B_LEN_T, GAME5B_CONTROL_DIM, 2, 2)

    with pytest.raises(ValueError, match="cover pixels at AR step 0"):
        encoder.forward(
            Game5BCtrlInput(image=None, control=control),
            autoregressive_index=0,
            cache=Game5BCtrlEncoderCache(vae={}),
        )
    # A rollout that never ran AR 0 has no latent to shape the zeros from.
    with pytest.raises(ValueError, match="before AR step 0"):
        encoder.forward(
            Game5BCtrlInput(image=None, control=control),
            autoregressive_index=2,
            cache=Game5BCtrlEncoderCache(vae={}),
        )


def test_negative_prompt_is_the_recipe_english_string() -> None:
    """This run was trained against the English negative prompt, not Wan's."""
    from eden.impl.game5b.pipeline import NEGATIVE_PROMPT

    assert NEGATIVE_PROMPT.startswith("Vibrant colors, overexposed, static")
    assert NEGATIVE_PROMPT.endswith("fused fingers, three legs")


def _bare_pipeline(encoder_factory):
    """A ``Game5BInferencePipeline`` without ``__init__`` (no real weights)."""
    from eden.impl.game5b.pipeline import Game5BInferencePipeline

    events: list[str] = []

    class _Pipeline(Game5BInferencePipeline):
        def _ensure_oneshot_encoders_loaded(self):
            if self.text_encoder is None:
                events.append("load")
                self.text_encoder = encoder_factory(events)

        def release_oneshot_encoders(self):
            events.append("release")
            self.text_encoder = None

        def _initialize_cache_from_embeddings(
            self, text_embeddings, image, *, height, width
        ):
            del height, width
            # The KV cache is allocated here; the encoder must already be gone.
            events.append("kv" if self.text_encoder is None else "kv-with-encoder")
            return SimpleNamespace(image=image, transformer_cache=SimpleNamespace())

    pipeline = Game5BInferencePipeline.__new__(_Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.diffusion_model = SimpleNamespace(
        config=SimpleNamespace(max_latent_frames=None), device=torch.device("cpu")
    )
    pipeline._text_cache = None
    pipeline._negative_text_cache = None
    pipeline.text_encoder = None
    return pipeline, events


class _Encoder(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def forward(self, text):  # noqa: D102 - test double
        self.events.append(f"encode:{text[0]}")
        return torch.zeros(1, 2)


def test_prompt_is_encoded_and_released_before_the_kv_cache() -> None:
    """UMT5 never coexists with the freshly allocated KV cache."""
    pipeline, events = _bare_pipeline(_Encoder)
    pipeline.initialize_cache(["a cat"], torch.zeros(1, 1, 3, 32, 32))
    assert events == ["load", "encode:a cat", "release", "kv"]


def test_restart_reuses_the_text_embeddings() -> None:
    """A second ``initialize_cache`` must not reload or re-run the encoder.

    The Game2V session rebuilds the rollout cache on every restart; reloading
    the released UMT5 and re-encoding an unchanged prompt is that stall.
    """
    pipeline, events = _bare_pipeline(_Encoder)
    image = torch.zeros(1, 1, 3, 32, 32)
    first = pipeline.initialize_cache(["a cat"], image)
    second = pipeline.initialize_cache(["a cat"], image)
    assert second is not first
    assert events.count("load") == 1 and events.count("encode:a cat") == 1
    # A prompt that actually changed is re-encoded (after a reload).
    pipeline.initialize_cache(["a dog"], image)
    assert events[-4:] == ["load", "encode:a dog", "release", "kv"]


def test_control_stream_resamples_onto_the_5b_latent_grid() -> None:
    """Features must land on pixels/16, not the 14B's pixels/8.

    A grid mismatch would be silently resampled a second time at consume time,
    so the ratio is pinned to the VAE's.
    """
    from eden.apps.game2v.game5b import make_game5b_control_stream

    built: dict[str, object] = {}

    class _Stream:
        def __init__(
            self, config, *, latent_height, latent_width, len_t, max_latent_frames
        ):
            built.update(
                config=config,
                latent_height=latent_height,
                latent_width=latent_width,
                len_t=len_t,
                max_latent_frames=max_latent_frames,
            )

    import eden.apps.game2v.game5b as game5b

    original = game5b.ProxyDinoControlStream
    game5b.ProxyDinoControlStream = _Stream  # type: ignore[assignment]
    try:
        make_game5b_control_stream(
            SimpleNamespace(
                model_video_width=GAME2V_5B_PIXEL_WIDTH,
                model_video_height=GAME2V_5B_PIXEL_HEIGHT,
                device="cpu",
                compile_control_encoder=False,
            ),
            SimpleNamespace(video_width=0, video_height=0),
        )
    finally:
        game5b.ProxyDinoControlStream = original  # type: ignore[assignment]

    assert built["latent_height"] == GAME2V_5B_PIXEL_HEIGHT // 16 == 44
    assert built["latent_width"] == GAME2V_5B_PIXEL_WIDTH // 16 == 80
    assert built["len_t"] == GAME2V_5B_LEN_T


def test_control_stream_rejects_an_unaligned_frame_size() -> None:
    """A frame size the VAE cannot halve four times is rejected up front."""
    from eden.apps.game2v.game5b import make_game5b_control_stream

    with pytest.raises(ValueError, match="divisible by the VAE"):
        make_game5b_control_stream(
            SimpleNamespace(
                model_video_width=1270,
                model_video_height=704,
                device="cpu",
                compile_control_encoder=False,
            ),
            SimpleNamespace(video_width=0, video_height=0),
        )
