# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Game2V end to end on the v0.2.0 runtime, with a stub model and a replayed proxy.

Drives ``run_session`` with the real ``Mp4ClientWindow``: the model loop's two
channels, the UI loop's list contract, the side-by-side canvas, the proxy
recording, the chunk log and the replay's condition-frame layout, all on the CPU.
"""

from __future__ import annotations

import json
import shutil
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from cam2v import Cam2VConditioning
from eden.apps.game2v.recording import iter_recording_frames
from eden.apps.game2v.session import Game2VSession, Game2VSessionConfig
from PIL import Image

from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.session_desc import (
    BackpressureMode,
    PresentationMode,
    SessionDesc,
)
from flashdreams.runtime_v2.session_runner import run_session

av = pytest.importorskip("av")

pytestmark = [
    pytest.mark.ci_cpu,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg on PATH"),
]

HEIGHT, WIDTH = 32, 48


class _StubPipeline:
    """Answers the pipeline calls the Game2V model loop makes; frame g is grey level g."""

    def __init__(self) -> None:
        self.prompts: list[list[str]] = []
        self.produced = 0

    def initialize_cache(self, text, image, *, release_oneshot_encoders=True):
        self.prompts.append(list(text))
        assert image.shape == (1, 3, HEIGHT, WIDTH)
        return object()

    def get_num_output_frames(self, autoregressive_index):
        return 9 if autoregressive_index == 0 else 12

    def generate(self, autoregressive_index, cache, input):
        count = self.get_num_output_frames(autoregressive_index)
        levels = torch.arange(self.produced, self.produced + count, dtype=torch.float32)
        self.produced += count
        return (
            (levels * 6 / 255 * 2 - 1)
            .view(-1, 1, 1, 1)
            .expand(count, 3, HEIGHT, WIDTH)
            .clone()
        )

    def finalize(self, autoregressive_index, cache):
        return {}


class _StubControl:
    """Records which capture frames each AR step's condition encode saw."""

    def __init__(self) -> None:
        self.pending: dict[int, object] = {}
        self.condition_ids: dict[int, list[int]] = {}

    def submit(self, autoregressive_index, pending_render):
        self.pending[autoregressive_index] = pending_render

    def take(self, autoregressive_index):
        pending = self.pending.pop(autoregressive_index)
        chunk: Any = pending.result() if isinstance(pending, Future) else pending
        self.condition_ids[autoregressive_index] = [
            frame.game_frame_id
            for frame in chunk.frames
            if frame.condition_frame is not None
        ]
        return torch.zeros(1)

    def reset(self):
        self.pending.clear()

    def close(self):
        self.reset()


def _write_video(path, frames, fps=24):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        stream.options = {"qscale": "1"}
        for index in range(frames):
            pixels = np.full((HEIGHT, WIDTH, 3), (index * 7) % 256, np.uint8)
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(pixels, format="rgb24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _decode(path):
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def test_replay_session_records_side_by_side_proxy_and_chunk_log(tmp_path):
    proxy_video = tmp_path / "proxy_in.mp4"
    _write_video(proxy_video, 36)
    cover = tmp_path / "cover.png"
    Image.new("RGB", (WIDTH, HEIGHT), (200, 100, 50)).save(cover)
    pipeline, control = _StubPipeline(), _StubControl()
    config = Game2VSessionConfig(
        conditioning=Cam2VConditioning(
            prompt="A replayed scene.",
            first_frame_path=cover,
            base_intrinsics=torch.tensor([24.0, 16.0, 24.0, 16.0]),
            world_scale=1.0,
        ),
        total_blocks=3,
        device=torch.device("cpu"),
        model_video_width=WIDTH,
        model_video_height=HEIGHT,
        first_frame_dtype=torch.float32,
        first_frame_interpolation="linear",
        warmup_blocks=0,
        control_stream_factory=lambda config, session_desc: control,
        proxy_video=proxy_video,
        proxy_frames=33,
        record_proxy=tmp_path / "proxy_out.mp4",
        chunk_log=tmp_path / "chunks.jsonl",
        side_by_side=True,
    )
    session_desc = SessionDesc(
        backpressure_mode=BackpressureMode.BLOCK,
        presentation_mode=PresentationMode.ON_DEMAND,
        frames_per_second_for_ui=240,
        frames_per_second_for_step=24,
        video_width=2 * WIDTH,
        video_height=HEIGHT,
    )
    output = tmp_path / "out.mp4"
    session = Game2VSession(pipeline=pipeline, config=config, session_desc=session_desc)
    assert run_session(session, Mp4ClientWindow(output), timeout_seconds=60) is None

    # One cache, built from the replay's prompt; condition frames at global 0, 4, 8, ...
    assert pipeline.prompts == [["A replayed scene."]]
    assert control.condition_ids == {0: [1, 5, 9], 1: [13, 17, 21], 2: [25, 29, 33]}

    expected_proxy = list(
        iter_recording_frames(
            proxy_video, height=HEIGHT, width=WIDTH, fps=24, num_frames=33
        )
    )
    recorded = _decode(tmp_path / "proxy_out.mp4")
    assert len(recorded) == 33
    canvas = _decode(output)
    assert len(canvas) == 33
    assert canvas[0].shape == (HEIGHT, 2 * WIDTH, 3)
    for index in range(33):
        proxy_level = int(expected_proxy[index][HEIGHT // 2, WIDTH // 2, 0])
        assert abs(int(recorded[index][HEIGHT // 2, WIDTH // 2, 0]) - proxy_level) <= 4
        assert abs(int(canvas[index][HEIGHT // 2, WIDTH // 2, 0]) - proxy_level) <= 4
        generated_level = int(canvas[index][HEIGHT // 2, WIDTH + WIDTH // 2, 0])
        assert abs(generated_level - 6 * index) <= 4, index

    lines = [
        json.loads(line)
        for line in (tmp_path / "chunks.jsonl").read_text().splitlines()
    ]
    assert [line["chunk"] for line in lines] == [0, 1, 2]
    assert [line["frames"] for line in lines] == [9, 12, 12]
    assert [line["first_frame"] for line in lines] == [0, 9, 21]
    assert lines[1]["condition_frames"] == [13, 17, 21]
    assert all(line["prompt"] == "A replayed scene." for line in lines)


def test_live_keys_flow_from_the_ui_loop_to_the_capture_worker():
    """The UI loop's key forward is what the capture worker drains between frames."""
    from eden.apps.game2v.session import _drain_live_game_events
    from numpy import uint64

    from flashdreams.runtime_v2.user_input_event import (
        KeyboardInputState,
        KeyboardUserInputEvent,
    )
    from flashdreams.runtime_v2.user_input_events import UserInputEvents

    session = Game2VSession(
        pipeline=object(),
        config=cast(
            Any,
            SimpleNamespace(
                device=torch.device("cpu"),
                postprocess_comparison=False,
                postprocess_enabled=False,
                pose_integrator_factory=__import__(
                    "eden.apps.game2v.controls", fromlist=["CameraPoseIntegrator"]
                ).CameraPoseIntegrator,
                game_slug="ember-den",
            ),
        ),
        session_desc=SessionDesc(),
    )
    session.init()
    try:
        ui = session.ui_loop
        ui._initialize_loop_state()
        events = UserInputEvents(
            [
                KeyboardUserInputEvent(
                    timestamp=uint64(1), key="f", state=KeyboardInputState.PRESSED
                ),
                KeyboardUserInputEvent(
                    timestamp=uint64(2), key="f", state=KeyboardInputState.RELEASED
                ),
            ]
        )
        assert ui.step(0, events) == []
        assert _drain_live_game_events(session.model_loop.state) == [
            {"action": "f", "phase": "down"},
            {"action": "f", "phase": "up"},
        ]
        session.model_loop.reset()
        assert _drain_live_game_events(session.model_loop.state) == []
    finally:
        session.close()
