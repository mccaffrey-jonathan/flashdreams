# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the proxy-DINO conditioned AR 14B Game2V preset."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from eden.apps.game2v.game14b import (
    game_cover_path,
    resolve_game14b_conditioning,
)
from eden.apps.game2v.game_renderer import registered_game_slugs
from eden.game14b_config import (
    GAME2V_LEN_T,
    PIPELINE_GAME2V_EDEN_WORLD_V2_14B,
    PIPELINE_GAME2V_EDEN_WORLD_V2_14B_TAEHV,
)
from eden.impl.game14b.checkpoint import game14b_dit_state_dict_from_diffusers
from eden.impl.game14b.control_stream import _stack_condition_frames
from eden.impl.game14b.network import (
    GAME14B_CONTROL_DIM,
    GAME14B_I2V_CHANNELS,
)

pytestmark = pytest.mark.ci_cpu


def test_channel_layout_matches_the_checkpoint_patch_embedding() -> None:
    """1060 input channels: 16 noisy + 4 mask + 16 condition + 1024 control."""
    network = PIPELINE_GAME2V_EDEN_WORLD_V2_14B.diffusion_model.transformer.network
    assert GAME14B_I2V_CHANNELS == 36
    assert network.in_dim == GAME14B_I2V_CHANNELS + GAME14B_CONTROL_DIM == 1060
    assert network.control_dim == GAME14B_CONTROL_DIM
    # Wan 2.1 I2V ships a CLIP vision branch; the recipe keeps it on.
    assert network.cross_attn_enable_img is True


def test_sampling_matches_the_sgf_v4_14b_schedule() -> None:
    """CFG off, 4-step distilled schedule at shift 3.0, sink 6 + window 12."""
    pipeline = PIPELINE_GAME2V_EDEN_WORLD_V2_14B
    transformer = pipeline.diffusion_model.transformer
    scheduler = pipeline.diffusion_model.scheduler
    model = pipeline.diffusion_model
    # CFG off: one conditional KV stream, the control encoded once per chunk.
    assert transformer.guidance_scale == 1.0
    assert transformer.len_t == GAME2V_LEN_T == 3
    # Wan 2.1 I2V channel concat composed with the clean-latent stamp.
    assert transformer.concat_image_mask_to_latent is True
    assert transformer.stamp_image_latent is True
    assert transformer.ti2v_first_frame_per_token_timestep is True
    # Cache-slot RoPE and fp32 AdaLN, the training numerics.
    assert transformer.network.apply_rope_before_kvcache is False
    assert transformer.network.fp32_adaln is True
    # One rank-0 noise draw broadcast to every shard, clean cache commits.
    assert model.noise_from_rank0 is True
    assert model.context_noise == 0
    # The shift-3 warped table yields the reference's [1000, 900, 750, 500].
    assert scheduler.num_inference_steps == 4
    assert scheduler.denoising_timesteps == [1000, 750, 500, 250]
    assert scheduler.shift == 3.0
    assert scheduler.sigma_max == 1.0
    assert scheduler.timestep_dtype == torch.float32
    assert scheduler.network_timestep_dtype is None
    # 18-frame attention budget: 6 fixed sink + 12 rolling.
    assert transformer.window_size_t == 12
    assert transformer.sink_size_t == 6
    assert transformer.sink_size_t + transformer.window_size_t == 18


class _FakeVAE:
    """Streaming-VAE stand-in: one distinct value per latent frame, counted globally."""

    temporal_compression_ratio = 4
    spatial_compression_ratio = 8

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []
        self.inputs: list[torch.Tensor] = []
        self.frames = 0

    def initialize_autoregressive_cache(self) -> dict:
        return {}

    def get_input_temporal_size(self, autoregressive_index: int, output: int) -> int:
        if autoregressive_index == 0:
            return 1 + (output - 1) * 4
        return output * 4

    def __call__(self, frames, *, autoregressive_index, cache):
        del cache
        count = int(frames.shape[-4])
        self.calls.append((autoregressive_index, count))
        self.inputs.append(frames.clone())
        latent_frames = (
            1 + (count - 1) // 4 if autoregressive_index == 0 else count // 4
        )
        values = torch.arange(
            self.frames, self.frames + latent_frames, dtype=frames.dtype
        )
        self.frames += latent_frames
        return (
            values.view(latent_frames, 1, 1, 1).expand(latent_frames, 16, 2, 2).clone()
        )


def _encoder(vae: _FakeVAE):
    from eden.impl.game14b.encoder import Game14BCtrlEncoder

    encoder = Game14BCtrlEncoder.__new__(Game14BCtrlEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.encoder = vae
    encoder._cover_cache = None
    return encoder


def test_condition_keeps_global_time_after_attention_window_fills() -> None:
    from eden.impl.game14b.encoder import Game14BCtrlEncoderCache, Game14BCtrlInput

    vae = _FakeVAE()
    encoder = _encoder(vae)
    cover = torch.ones(1, 3, 8, 8)
    control = torch.zeros(3, 1024, 2, 2)
    cache = Game14BCtrlEncoderCache(vae={})
    out = encoder(Game14BCtrlInput(image=cover, control=control), 0, cache)
    assert vae.calls == [(0, 1), (0, 9)]
    assert torch.equal(vae.inputs[1][:1], cover)
    assert torch.all(vae.inputs[1][1:] == 0)
    assert out.i2v.latent[:, 0, 0, 0].tolist() == [0.0, 2.0, 3.0]
    assert out.i2v.mask[:, 0, 0, 0].tolist() == [1.0, 0.0, 0.0]
    for index in range(1, 8):
        out = encoder(Game14BCtrlInput(image=None, control=control), index, cache)
        assert out.i2v.latent[:, 0, 0, 0].tolist() == [
            float(1 + index * 3 + offset) for offset in range(3)
        ]
        assert torch.all(out.i2v.mask == 0)
    assert len(vae.calls) == 9
    restart = Game14BCtrlEncoderCache(vae={})
    encoder(Game14BCtrlInput(image=cover, control=control), 0, restart)
    # Cover is reused; the grey stream starts from its own fresh rollout cache.
    assert vae.calls[-1] == (0, 9)
    assert len(vae.calls) == 10


def test_later_chunk_before_ar0_is_rejected() -> None:
    from eden.impl.game14b.encoder import Game14BCtrlEncoderCache, Game14BCtrlInput

    encoder = _encoder(_FakeVAE())
    with pytest.raises(ValueError, match="before AR"):
        encoder(
            Game14BCtrlInput(image=None, control=torch.zeros(3, 1024, 1, 1)),
            2,
            Game14BCtrlEncoderCache(vae={}),
        )


def test_restart_reuses_the_one_shot_encodings() -> None:
    """A second ``initialize_cache`` must not reload or re-run the encoders."""
    from eden.impl.game14b.encoder import Game14BCtrlEncoderCache
    from eden.impl.game14b.pipeline import Game14BInferencePipeline

    calls = {"base": 0}

    class _Pipeline(Game14BInferencePipeline):
        def _initialize_cache_recording(self, text, image, **kwargs):
            del kwargs
            calls["base"] += 1
            self._text_cache = (tuple(text), torch.zeros(1, 2))
            self._image_cache = (image, torch.zeros(1, 2))
            return SimpleNamespace(
                image=image,
                encoder_cache=Game14BCtrlEncoderCache(vae={}),
                transformer_cache=SimpleNamespace(),
            )

        def _initialize_cache_from_embeddings(
            self, text_embeddings, image, image_embeddings, *, height, width
        ):
            del height, width
            if text_embeddings is None or image_embeddings is None:
                return None
            return SimpleNamespace(
                image=image,
                encoder_cache=Game14BCtrlEncoderCache(vae={}),
                transformer_cache=SimpleNamespace(),
            )

    pipeline = Game14BInferencePipeline.__new__(_Pipeline)
    pipeline.diffusion_model = SimpleNamespace(
        config=SimpleNamespace(max_latent_frames=None)
    )
    pipeline._text_cache = None
    pipeline._negative_text_cache = None
    pipeline._image_cache = None
    pipeline._pending_text = None
    pipeline._pending_image = None
    image = torch.zeros(1, 1, 3, 8, 8)

    first = pipeline.initialize_cache(["a cat"], image)
    second = pipeline.initialize_cache(["a cat"], image)
    assert calls == {"base": 1}
    assert second is not first
    assert second.encoder_cache is not first.encoder_cache
    pipeline.initialize_cache(["a dog"], image)
    assert calls == {"base": 2}


def test_taehv_variant_swaps_only_the_decoder() -> None:
    """LightTAE matches the Wan VAE's compression, so pacing is unchanged."""
    from flashdreams.recipes.taehv import TeahvVAEDecoder, TeahvVAEDecoderConfig
    from flashdreams.recipes.wan.autoencoder.vae import WanVAE

    base = PIPELINE_GAME2V_EDEN_WORLD_V2_14B
    variant = PIPELINE_GAME2V_EDEN_WORLD_V2_14B_TAEHV
    assert isinstance(variant.decoder, TeahvVAEDecoderConfig)
    # The ratios live on the decoder implementations, and they must agree or
    # the renderer's capture pacing would drift from the decoded frames.
    assert (
        TeahvVAEDecoder.TEMPORAL_COMPRESSION_RATIO
        == WanVAE.TEMPORAL_COMPRESSION_RATIO
        == 4
    )
    assert (
        TeahvVAEDecoder.SPATIAL_COMPRESSION_RATIO
        == WanVAE.SPATIAL_COMPRESSION_RATIO
        == 8
    )
    # Sampling and the attention window are inherited unchanged.
    assert variant.diffusion_model.transformer.len_t == 3
    assert variant.diffusion_model.transformer.window_size_t == 12
    assert variant.diffusion_model.scheduler.num_inference_steps == 4


def test_negative_prompt_is_the_14b_string() -> None:
    """The 14B CFG text: the T2V string with a leading ``镜头晃动，``.

    The camera model's v2 prompt must not leak in, and the shared recipe
    constant other Wan pipelines use stays untouched.
    """
    from flashdreams.recipes.wan.transformer.constants import NEGATIVE_PROMPT

    from eden.impl.game14b.pipeline import Game14BInferencePipeline
    from eden.impl.pipeline import EDEN_WORLD_V2_NEGATIVE_PROMPT

    negative = Game14BInferencePipeline._negative_prompt()
    assert negative == "镜头晃动，" + NEGATIVE_PROMPT
    assert not NEGATIVE_PROMPT.startswith("镜头晃动")
    assert negative != EDEN_WORLD_V2_NEGATIVE_PROMPT


@pytest.mark.parametrize("slug", registered_game_slugs())
def test_first_frame_comes_from_each_games_own_cover(slug: str) -> None:
    """Every game conditions on its own cover, not one shared image."""
    cover = game_cover_path(slug)
    assert cover.is_file()
    assert cover.parent.name == slug

    conditioning = resolve_game14b_conditioning(
        {"game": slug, "pixel_width": 832, "pixel_height": 480}
    )
    assert conditioning.first_frame_path == cover
    # The prompt defaults to the registry's static scene description.
    assert conditioning.prompt
    assert len(conditioning.prompt) > 40


def test_covers_differ_between_games() -> None:
    slugs = registered_game_slugs()
    assert {"ember-den", "jade-duel"} <= set(slugs)
    assert len({game_cover_path(slug) for slug in slugs}) == len(slugs)
    prompts = {
        slug: resolve_game14b_conditioning({"game": slug}).prompt for slug in slugs
    }
    assert len(set(prompts.values())) == len(slugs)


def test_explicit_image_and_prompt_override_the_registry(tmp_path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"not really a png, only the path is read here")
    conditioning = resolve_game14b_conditioning(
        {"game": "ember-den", "image_path": image, "prompt": "a custom prompt"}
    )
    assert conditioning.first_frame_path == image
    assert conditioning.prompt == "a custom prompt"


def test_unregistered_game_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unregistered game"):
        resolve_game14b_conditioning({"game": "no-such-game"})


def test_checkpoint_remap_covers_the_i2v_and_clip_branches() -> None:
    """The three key groups the shared Diffusers table does not handle."""
    source = {
        "blocks.0.attn2.add_k_proj.weight": torch.empty(0),
        "blocks.0.attn2.add_v_proj.weight": torch.empty(0),
        "blocks.0.attn2.norm_added_k.weight": torch.empty(0),
        "condition_embedder.image_embedder.norm1.weight": torch.empty(0),
        "condition_embedder.image_embedder.ff.net.0.proj.weight": torch.empty(0),
        "condition_embedder.image_embedder.ff.net.2.weight": torch.empty(0),
        "condition_embedder.image_embedder.norm2.weight": torch.empty(0),
        "patch_embedding.condition_norm.weight": torch.empty(0),
    }
    remapped = game14b_dit_state_dict_from_diffusers(source)
    assert set(remapped) == {
        "blocks.0.cross_attn.k_img.weight",
        "blocks.0.cross_attn.v_img.weight",
        "blocks.0.cross_attn.norm_k_img.weight",
        # MLPProj: LayerNorm / Linear / GELU / Linear / LayerNorm -> 0,1,3,4.
        "img_emb.proj.0.weight",
        "img_emb.proj.1.weight",
        "img_emb.proj.3.weight",
        "img_emb.proj.4.weight",
        # The control RMSNorm is already in FlashDreams layout.
        "patch_embedding.condition_norm.weight",
    }


def test_checkpoint_remap_still_applies_the_shared_table() -> None:
    remapped = game14b_dit_state_dict_from_diffusers(
        {
            "blocks.0.attn1.to_q.weight": torch.empty(0),
            "blocks.0.norm2.weight": torch.empty(0),
            "blocks.0.scale_shift_table": torch.empty(0),
            "scale_shift_table": torch.empty(0),
            "proj_out.weight": torch.empty(0),
        }
    )
    assert set(remapped) == {
        "blocks.0.self_attn.q.weight",
        # Diffusers' affine norm2 is FlashDreams' cross-attention norm3.
        "blocks.0.norm3.weight",
        "blocks.0.modulation",
        "head.modulation",
        "head.head.weight",
    }


def test_condition_frames_are_stacked_in_zero_to_one_rgb() -> None:
    """One condition render per latent frame, normalized to [0, 1]."""
    import numpy as np

    frames = [
        SimpleNamespace(
            condition_frame=(
                np.full((480, 832, 3), 255, dtype=np.uint8) if index % 4 == 0 else None
            )
        )
        for index in range(12)
    ]
    stacked = _stack_condition_frames(frames, expected=3)
    assert stacked.shape == (3, 3, 480, 832)
    assert stacked.dtype == torch.float32
    assert stacked.min() >= 0.0 and stacked.max() <= 1.0
    assert torch.allclose(stacked, torch.ones_like(stacked))


def test_condition_frame_count_must_match_the_chunk() -> None:
    import numpy as np

    frames = [
        SimpleNamespace(condition_frame=np.zeros((4, 4, 3), dtype=np.uint8))
        for _ in range(2)
    ]
    with pytest.raises(RuntimeError, match="condition frames"):
        _stack_condition_frames(frames, expected=3)
