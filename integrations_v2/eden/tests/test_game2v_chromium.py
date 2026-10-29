# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Chromium capture of every registered EdenLive game (no model, no weights).

Manual: needs ``EDEN_GAMES_ROOT`` (with ``npm ci`` run in it), Playwright and
``python -m playwright install chromium``::

    EDEN_GAMES_ROOT=.../realtime/games pytest integrations_v2/eden/tests/test_game2v_chromium.py -m manual
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.manual

pytest.importorskip("playwright.sync_api")


def _slugs():
    from eden.apps.game2v.game_renderer import registered_game_slugs

    return registered_game_slugs() or ("<EDEN_GAMES_ROOT unset>",)


@pytest.mark.parametrize("slug", _slugs())
def test_capture_renders_model_sized_frames_and_the_autopilot_plays(
    slug, real_games_root
):
    from eden.apps.game2v.game_renderer import (
        _SOFTWARE_RENDERER_MARKS,
        GameCaptureWorker,
        default_autopilot_script,
    )
    from eden.apps.game2v.game_timeline import GameTimeline

    worker = GameCaptureWorker(
        slug, width=1280, height=704, autopilot=default_autopilot_script()
    )
    timeline = GameTimeline(slug)
    try:
        chunks = [
            worker.submit(timeline.begin_chunk(9 if index == 0 else 12, 24)).result(
                timeout=120
            )
            for index in range(8)
        ]
    finally:
        worker.close()
    print(f"{slug}: WebGL renderer {worker.webgl_renderer!r}")
    assert not any(
        mark in worker.webgl_renderer for mark in _SOFTWARE_RENDERER_MARKS
    ), f"{slug} rendered on {worker.webgl_renderer}; expected the GPU"
    frames = [frame for chunk in chunks for frame in chunk.frames]
    assert len(frames) == 9 + 7 * 12
    assert all(frame.ui_frame.shape == (704, 1280, 3) for frame in frames)
    assert sum(frame.condition_frame is not None for frame in frames) == 3 * len(chunks)
    # The autopilot plays: the picture and the game state move.
    assert frames[0].ui_frame.tobytes() != frames[-1].ui_frame.tobytes()
    assert frames[-1].ui_state != frames[0].ui_state
