# SPDX-FileCopyrightText: Copyright (c) 2026 Vast AI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-compatible noise layout of :class:`EdenDiffusionModel`.

The EdenLive reference (VAST offline package, LightX2V's runner) draws one
whole-video initial noise tensor in ``(C, T, H, W)`` order, then one re-noise
tensor per scheduler step, all from a single CUDA generator. These tests pin
that stream on the CPU with a pass-through transformer.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from eden.impl.diffusion import EdenDiffusionModel

pytestmark = pytest.mark.ci_cpu


class _PassThroughTransformer(torch.nn.Module):
    config = SimpleNamespace(len_t=3)
    device = torch.device("cpu")
    dtype = torch.float32

    def unpatchify_and_maybe_gather_cp(self, value):
        return value

    def patchify_and_maybe_split_cp(self, value):
        return value


def _bare_model(**config) -> EdenDiffusionModel:
    model = EdenDiffusionModel.__new__(EdenDiffusionModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(seed=2026092901, **config)
    model.transformer = _PassThroughTransformer()
    model._rng = None
    model._initial_video_noise = None
    return model


def test_video_noise_draw_preserves_channels_first_rng_and_cpu_storage() -> None:
    model = _bare_model(max_latent_frames=9)
    generator = torch.Generator().manual_seed(model.config.seed)
    expected = torch.randn(2, 9, 2, 2, generator=generator)  # (C, T, H, W)
    chunks = [
        model._video_initial_noise(torch.empty(3, 2, 2, 2), index) for index in range(3)
    ]
    # FlashDreams lays latents out (T, C, H, W); the draw was (C, T, H, W).
    assert torch.equal(torch.cat(chunks).movedim(-3, -4), expected)
    assert model._initial_video_noise is not None
    assert model._initial_video_noise.device.type == "cpu"
    assert torch.equal(model.rng.get_state(), generator.get_state())
    # The re-noise continues the same stream, again channels-first.
    renoise = model._rank0_noise(torch.empty(3, 2, 2, 2))
    expected_renoise = torch.randn(2, 3, 2, 2, generator=generator).movedim(-3, -4)
    assert torch.equal(renoise, expected_renoise)


def test_video_noise_rejects_reading_past_the_bounded_length() -> None:
    model = _bare_model(max_latent_frames=6)
    model._video_initial_noise(torch.empty(3, 2, 2, 2), 1)
    with pytest.raises(ValueError, match="max_latent_frames"):
        model._video_initial_noise(torch.empty(3, 2, 2, 2), 2)


def test_reset_noise_rewinds_the_stream() -> None:
    model = _bare_model(max_latent_frames=3)
    first = model._video_initial_noise(torch.empty(3, 2, 2, 2), 0).clone()
    model._rank0_noise(torch.empty(3, 2, 2, 2))
    model.reset_noise()
    assert torch.equal(model._video_initial_noise(torch.empty(3, 2, 2, 2), 0), first)
