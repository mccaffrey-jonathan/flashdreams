# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from eden.apps.game2v.controls import CameraPoseIntegrator
from eden.apps.game2v.game_hud import envelope
from eden.apps.game2v.game_ui import Game2VUILoop
from eden.apps.game2v.session import Game2VSession
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize('use_ui', [True, False])
def test_game2v_registers_paired_ui_without_renderer_constructor_args(use_ui):
    config = SimpleNamespace(
        device=torch.device('cpu'),
        postprocess_comparison=False,
        postprocess_enabled=False,
        pose_integrator_factory=CameraPoseIntegrator,
        game_slug='freeze',
    )
    session = Game2VSession(pipeline=object(), config=config,
                           session_desc=SessionDesc(), use_ui=use_ui)
    session.init()
    assert isinstance(session.ui_loop, Game2VUILoop)
    assert session.ui_loop.session_desc == session.session_desc
    assert session.ui_loop._last_count == 0
    session.close()


def test_ui_loop_forwards_presented_game_metadata():
    loop = Game2VUILoop.__new__(Game2VUILoop)
    loop.output_layout = VideoTensorLayout.tchw
    loop._last_count = 0
    loop._prompt_chunk = None
    model = torch.zeros(3, 2, 2)
    game = torch.ones(3, 2, 2)
    meta = envelope('freeze', 4, {'moves': 2})
    loop.presented_model_frames = lambda: (model, game)
    loop.presented_model_metadata = lambda channel: meta if channel == 1 else None
    loop._presentation_manager = SimpleNamespace(
        presented_frame_count=1, presented_client_messages=lambda channel=0: ())
    result = loop.step(0, events=None)
    assert result is not None
    assert result.client_messages == (meta,)
    torch.testing.assert_close(result.auxiliary_frames['game'], game)
    assert loop.step(1, events=None) is None


def test_ui_loop_sends_the_chunk_prompt_once():
    """Lines go out on a chunk's first frame and stay up until the next chunk."""
    loop = Game2VUILoop.__new__(Game2VUILoop)
    loop.output_layout = VideoTensorLayout.tchw
    loop._last_count = 0
    loop._prompt_chunk = None
    prompt = {
        'type': 'prompt', 'static': 'A quiet courtyard.',
        'events': ['The keeper steps left.'],
    }
    manager = SimpleNamespace(
        presented_frame_count=1,
        presented_chunk_count=1,
        presented_client_messages=lambda channel=0: (prompt,),
    )
    loop.presented_model_frames = lambda: (torch.zeros(3, 2, 2),)
    loop.presented_model_metadata = lambda channel: None
    loop._presentation_manager = manager

    first = loop.step(0, events=None)
    assert first.client_messages == (prompt,)

    # Later frames of the same chunk leave the panel alone, including the last
    # one: nothing is blanked between chunks.
    for step, frame_count in enumerate((2, 3), start=1):
        manager.presented_frame_count = frame_count
        assert loop.step(step, events=None).client_messages == ()

    # The next chunk replaces the box with its own set.
    manager.presented_frame_count = 4
    manager.presented_chunk_count = 2
    manager.presented_client_messages = lambda channel=0: ({
        'type': 'prompt', 'static': 'A quiet courtyard.',
        'events': ['A crate slides.'],
    },)
    replacement = loop.step(3, events=None).client_messages
    assert replacement[0]['events'] == ['A crate slides.']
    assert replacement[0]['static'] == 'A quiet courtyard.'


def test_ui_loop_omits_messages_when_the_game_channel_has_no_metadata():
    loop = Game2VUILoop.__new__(Game2VUILoop)
    loop.output_layout = VideoTensorLayout.tchw
    loop._last_count = 0
    loop._prompt_chunk = None
    loop.presented_model_frames = lambda: (torch.zeros(3, 2, 2), torch.ones(3, 2, 2))
    loop.presented_model_metadata = lambda channel: None
    loop._presentation_manager = SimpleNamespace(
        presented_frame_count=3, presented_client_messages=lambda channel=0: ())
    result = loop.step(0, events=None)
    assert result is not None
    assert result.client_messages == ()
    assert result.auxiliary_frames['game'].shape == (3, 2, 2)
