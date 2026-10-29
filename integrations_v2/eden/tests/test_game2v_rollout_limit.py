# SPDX-FileCopyrightText: Copyright (c) 2026 Vast AI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

"""Game2V rejects rollouts that exhaust the preallocated video noise."""

from dataclasses import replace

import pytest
from eden.apps.game2v.adapter import (
    create_app_v2_5b_taehv,
    create_app_v2_14b_taehv,
)

from flashdreams.infra.config import derive_config

pytestmark = pytest.mark.ci_cpu


@pytest.fixture(params=[create_app_v2_5b_taehv, create_app_v2_14b_taehv])
def app(request, fake_games_root):
    return request.param()


def _bounded(app, max_latent_frames):
    app.defaults = replace(
        app.defaults,
        pipeline_config=derive_config(
            app.pipeline_config,
            diffusion_model={"max_latent_frames": max_latent_frames},
        ),
    )


def test_live_rollout_is_unbounded_by_default(app):
    """A live game has no known length, so its noise is drawn per chunk."""
    app.init(["--device", "cpu"])
    assert app._total_blocks == app.defaults.total_blocks == 20
    assert app.pipeline_config.diffusion_model.max_latent_frames is None


@pytest.mark.parametrize("blocks", [1, 120])
def test_bounded_rollout_accepts_valid_chunk_counts(app, blocks):
    _bounded(app, 360)
    app.init(["--device", "cpu", "--total-blocks", str(blocks)])
    assert app._total_blocks == blocks


@pytest.mark.parametrize("blocks", [121, 10000])
def test_bounded_rollout_rejects_overrun_before_loading_weights(app, blocks):
    _bounded(app, 360)
    with pytest.raises(ValueError, match=r"--total-blocks must be <= 120"):
        app.init(["--device", "cpu", "--total-blocks", str(blocks)])
    assert app._pipeline is None


def test_partial_last_chunk_counts_as_a_valid_block(app):
    _bounded(app, 7)
    app.init(["--device", "cpu", "--total-blocks", "3"])
    assert app._total_blocks == 3
    with pytest.raises(ValueError, match=r"--total-blocks must be <= 3"):
        app.init(["--device", "cpu", "--total-blocks", "4"])


def test_unbounded_noise_keeps_long_rollouts_available(app):
    app.init(["--device", "cpu", "--total-blocks", "10000"])
    assert app._total_blocks == 10000
