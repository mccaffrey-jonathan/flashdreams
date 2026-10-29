# SPDX-FileCopyrightText: Copyright (c) 2026 Vast AI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

"""Game2V session wiring and the v0.2.0 UI-loop contract (list returns, channels)."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from eden.apps.game2v import extensions
from eden.apps.game2v.controls import CameraPoseIntegrator
from eden.apps.game2v.game_ui import Game2VUILoop, side_by_side_frame
from eden.apps.game2v.session import Game2VSession

from flashdreams.api_v2.loop import ModelInferenceState
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


def _session(**config):
    return Game2VSession(
        pipeline=object(),
        config=cast(
            Any,
            SimpleNamespace(
                device=torch.device("cpu"),
                postprocess_comparison=False,
                postprocess_enabled=False,
                pose_integrator_factory=CameraPoseIntegrator,
                game_slug="ember-den",
                **config,
            ),
        ),
        session_desc=SessionDesc(),
    )


def test_session_shares_one_key_queue_between_ui_and_model_loops():
    session = _session()
    session.init()
    try:
        assert isinstance(session.ui_loop, Game2VUILoop)
        assert session.ui_loop.session_desc == session.session_desc
        assert session.ui_loop._last_count == 0
        assert session.ui_loop._live_input is session.model_loop.state.live_input
        assert session.ui_loop._side_by_side is False
        # No browser window, no panel.
        assert "game2v" not in extensions._extensions
    finally:
        session.close()


def test_panel_mounts_only_for_the_integration_browser_window():
    session = _session(register_panel=True, side_by_side=True)
    session.init()
    try:
        assert "game2v" in extensions._extensions
        assert session.ui_loop._side_by_side is True
    finally:
        session.close()
    assert "game2v" not in extensions._extensions


class _Presented:
    """Stand-in for the presentation manager behind the UI loop's accessors."""

    def __init__(self):
        self.frames = ()
        self.count = 0
        self.pending = False


def _loop(*, side_by_side=False):
    loop: Any = Game2VUILoop(side_by_side=side_by_side)
    loop._initialize_loop_state()
    loop.state = None
    loop.output_layout = VideoTensorLayout.tchw
    presented = _Presented()
    loop.presented_model_frames = lambda: presented.frames
    loop.has_pending_model_frames = lambda: presented.pending
    loop._presentation_manager = SimpleNamespace()
    return loop, presented


@pytest.fixture
def count_accessor(monkeypatch):
    """Back ``presented_model_frame_count`` with the stand-in's counter."""
    monkeypatch.setattr(
        Game2VUILoop,
        "presented_model_frame_count",
        property(lambda self: self._test_presented.count),
    )


def test_ui_loop_returns_a_list_with_the_generated_channel_once(count_accessor):
    loop, presented = _loop()
    loop._test_presented = presented
    assert loop.step(0, events=None) == []

    generated = torch.zeros(3, 2, 2)
    proxy = torch.full((3, 2, 2), 255, dtype=torch.uint8)
    presented.frames, presented.count = (generated, proxy), 1
    results = loop.step(1, events=None)
    assert len(results) == 1
    # The proxy rides on channel 1 for windows and recorders; it is not drawn
    # over the generated frame.
    torch.testing.assert_close(results[0].read_output(), generated.unsqueeze(0))
    assert results[0].frame_count == 1
    # The same presented frame is not emitted twice (an MP4 would duplicate it).
    assert loop.step(2, events=None) == []
    presented.count = 2
    assert len(loop.step(3, events=None)) == 1


def test_ui_loop_side_by_side_puts_the_proxy_left_in_model_range(count_accessor):
    loop, presented = _loop(side_by_side=True)
    loop._test_presented = presented
    generated = torch.full((3, 4, 6), 0.5)
    proxy = torch.full((3, 4, 6), 255, dtype=torch.uint8)
    presented.frames, presented.count = (generated, proxy), 1
    output = loop.step(0, events=None)[0].read_output()
    assert output.shape == (1, 3, 4, 12)
    torch.testing.assert_close(output[0, :, :, :6], torch.ones(3, 4, 6))
    torch.testing.assert_close(output[0, :, :, 6:], generated)


def test_side_by_side_without_a_proxy_channel_is_black_left():
    frame = side_by_side_frame(None, torch.zeros(3, 2, 2))
    assert frame.shape == (3, 2, 4)
    assert torch.all(frame[:, :, :2] == -1)


def test_side_by_side_resizes_a_proxy_of_another_size():
    frame = side_by_side_frame(
        torch.zeros(3, 8, 8, dtype=torch.uint8), torch.zeros(3, 4, 4)
    )
    assert frame.shape == (3, 4, 8)


def test_ui_loop_finishes_after_the_last_presented_frame(count_accessor, monkeypatch):
    loop, presented = _loop()
    loop._test_presented = presented
    state = {"value": ModelInferenceState.RUNNING}
    monkeypatch.setattr(
        Game2VUILoop, "model_inference_state", property(lambda self: state["value"])
    )
    presented.frames, presented.count = (torch.zeros(3, 2, 2),), 1
    loop.step(0, events=None)
    assert not loop.is_finished()
    state["value"] = ModelInferenceState.FINISHED
    presented.pending = True
    assert not loop.is_finished()
    presented.pending = False
    assert loop.is_finished()
    presented.count = 2
    assert not loop.is_finished()
