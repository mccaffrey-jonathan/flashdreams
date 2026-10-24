# SPDX-FileCopyrightText: Copyright (c) 2026 Vast AI
# SPDX-License-Identifier: Apache-2.0

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
def app(request):
    return request.param()


def test_default_rollout_stops_at_the_model_limit(app):
    app.init(['--device', 'cpu'])
    assert app._total_blocks == 120
    assert app.pipeline_config.diffusion_model.max_latent_frames == 360


@pytest.mark.parametrize('blocks', [1, 120])
def test_bounded_rollout_accepts_valid_chunk_counts(app, blocks):
    app.init(['--device', 'cpu', '--total-blocks', str(blocks)])
    assert app._total_blocks == blocks


@pytest.mark.parametrize('blocks', [121, 10000])
def test_bounded_rollout_rejects_overrun_before_loading_weights(app, blocks):
    with pytest.raises(ValueError, match=r'--total-blocks must be <= 120'):
        app.init(['--device', 'cpu', '--total-blocks', str(blocks)])
    assert app._pipeline is None


def test_partial_last_chunk_counts_as_a_valid_block(app):
    app.defaults = replace(
        app.defaults,
        pipeline_config=derive_config(
            app.pipeline_config, diffusion_model={'max_latent_frames': 7}
        ),
    )
    app.init(['--device', 'cpu', '--total-blocks', '3'])
    assert app._total_blocks == 3
    with pytest.raises(ValueError, match=r'--total-blocks must be <= 3'):
        app.init(['--device', 'cpu', '--total-blocks', '4'])


def test_unbounded_noise_keeps_long_rollouts_available(app):
    app.defaults = replace(
        app.defaults,
        pipeline_config=derive_config(
            app.pipeline_config, diffusion_model={'max_latent_frames': None}
        ),
    )
    app.init(['--device', 'cpu', '--total-blocks', '10000'])
    assert app._total_blocks == 10000
