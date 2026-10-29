# SPDX-FileCopyrightText: Copyright (c) 2026 Vast AI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

import base64
import json
import threading
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from eden.apps.game2v import game_renderer
from eden.apps.game2v.game_hud import envelope
from eden.apps.game2v.game_renderer import (
    DEFAULT_CAPTURE_HEIGHT,
    DEFAULT_CAPTURE_WIDTH,
    GameCaptureWorker,
    _chunk_capture_request,
    _decode_data_url,
    _expand_captured_frames,
    _sleep_until_frame_deadline,
)
from eden.apps.game2v.game_timeline import GameTimeline
from eden.apps.game2v.session import (
    Game2VModelState,
    _begin_game_chunk,
    _collect_game_chunk,
    _discard_pending_game_chunk,
    _drain_live_game_events,
    _game_input_payloads,
    _pace_duration_s,
)
from PIL import Image

pytestmark = pytest.mark.ci_cpu


def test_captured_ticks_follow_model_fps_not_a_fixed_24():
    """Sim stays 60 Hz; captured-frame spacing is the model's fps."""
    for fps, expected_end in ((16, 45), (24, 30), (8, 90)):
        timeline = GameTimeline("freeze")
        chunk = timeline.begin_chunk(12, fps)
        assert chunk.simulation_tick_end == expected_end
        ticks = [frame.simulation_tick for frame in chunk.frames]
        assert ticks == [round((index + 1) / fps * 60) for index in range(12)]
        # Capture dt in the page is (tick - previous) / 60, so a 16 fps model
        # gets ~1/16 s of visual interpolation, not a hardcoded 1/24.
        deltas = [
            ticks[0],
            *(later - earlier for earlier, later in zip(ticks, ticks[1:])),
        ]
        assert sum(deltas) == expected_end
        assert sum(deltas) / len(deltas) == pytest.approx(60 / fps)


def test_condition_samples_align_with_latent_stride():
    """Each chunk carries one condition frame per latent frame, at a fixed offset.

    The sample positions are relative to the chunk, not to the running frame
    count. A causal first chunk is 9 frames and later chunks are 12, neither
    divisible by the stride, so a global counter would shift the phase by one
    frame every chunk and a restarted rollout would open a frame short.
    """
    timeline = GameTimeline("freeze")
    first = timeline.begin_chunk(9, 16)
    second = timeline.begin_chunk(12, 16)

    assert [
        frame.game_frame_id for frame in first.frames if frame.capture_condition
    ] == [1, 5, 9]
    assert [
        frame.game_frame_id for frame in second.frames if frame.capture_condition
    ] == [10, 14, 18]
    assert [frame.capture_condition for frame in first.frames] == [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ]


def test_condition_stride_survives_reset():
    timeline = GameTimeline("freeze", condition_stride=4)
    timeline.begin_chunk(12, 24)
    timeline.reset()
    chunk = timeline.begin_chunk(12, 24)
    assert [
        frame.game_frame_id for frame in chunk.frames if frame.capture_condition
    ] == [1, 5, 9]


def test_restarted_rollout_keeps_a_full_condition_set():
    """A restart rewinds the model but not the rendered frame count.

    The new round opens at autoregressive index zero, whose chunk is the
    9-frame causal one. Its condition frames must still be three, exactly as
    the first round's were, even though the timeline was not reset.
    """
    timeline = GameTimeline("freeze")
    for index in range(14):
        timeline.begin_chunk(9 if index == 0 else 12, 16)
    # The in-flight successor is discarded on restart, then the round reopens.
    reopened = timeline.begin_chunk(9, 16)
    assert [frame.capture_condition for frame in reopened.frames] == [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ]


def test_decode_data_url_returns_rgb():
    rgb = _decode_data_url(
        _jpeg_data_url(np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8))
    )
    assert rgb.shape == (1, 2, 3)
    assert rgb.dtype == np.uint8


def test_capture_request_renders_every_frame():
    timeline = GameTimeline("freeze")
    chunk = timeline.begin_chunk(8, 16)
    request = _chunk_capture_request(chunk)
    assert request["events"] == []
    assert [frame["targetTick"] for frame in request["frames"]] == [
        packet.simulation_tick for packet in chunk.frames
    ]
    assert all(
        "captureCondition" not in frame and "captureStream" not in frame
        for frame in request["frames"]
    )


def test_expand_decodes_every_ui_frame_and_conditions_on_the_latent_subset():
    timeline = GameTimeline("freeze")
    chunk = timeline.begin_chunk(9, 16)
    frames = _expand_captured_frames(chunk, _stub_capture(chunk))
    assert [np.asarray(frame.ui_frame)[0, 0, 0] for frame in frames] == [
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
    ]
    assert [
        index for index, frame in enumerate(frames) if frame.condition_frame is not None
    ] == [0, 4, 8]
    assert all(
        frame.condition_frame is frame.ui_frame
        for frame in frames
        if frame.capture_condition
    )
    assert [frame.ui_state for frame in frames] == [
        {"moves": index} for index in range(9)
    ]


def test_expand_rejects_frame_count_mismatch():
    timeline = GameTimeline("freeze")
    chunk = timeline.begin_chunk(4, 16)
    with pytest.raises(RuntimeError, match="3 frames"):
        _expand_captured_frames(chunk, _stub_capture(chunk)[:3])


def test_hud_envelope_forwards_state_verbatim():
    payload = envelope("freeze", 12, {"moves": 3, "helpOpen": True})
    assert payload == {
        "type": "game_ui",
        "slug": "freeze",
        "frame_id": 12,
        "state": {"moves": 3, "helpOpen": True},
    }
    assert envelope("tracer", 1, None)["state"] is None


def _key(key, pressed, timestamp=1):
    from numpy import uint64

    from flashdreams.runtime_v2.user_input_event import (
        KeyboardInputState,
        KeyboardUserInputEvent,
    )

    return KeyboardUserInputEvent(
        timestamp=uint64(timestamp),
        key=key,
        state=KeyboardInputState.PRESSED if pressed else KeyboardInputState.RELEASED,
    )


def test_game_input_payloads_lowercase_keyboard_edges_only():
    from numpy import uint64

    from flashdreams.runtime_v2.user_input_event import FocusUserInputEvent
    from flashdreams.runtime_v2.user_input_events import UserInputEvents

    payloads = _game_input_payloads(
        UserInputEvents(
            [
                _key("W", True, 1),
                FocusUserInputEvent(timestamp=uint64(2), focused=False),
                _key("ArrowLeft", False, 3),
            ]
        ),
    )
    assert payloads == [
        {"action": "w", "phase": "down"},
        {"action": "arrowleft", "phase": "up"},
    ]


def test_ui_loop_forwards_keys_to_the_capture_worker_without_the_model_reader():
    """Keys reach the in-flight render through the UI loop, between model steps."""
    import queue

    from eden.apps.game2v.game_ui import Game2VUILoop

    from flashdreams.runtime_v2.event_buffer import EventBuffer
    from flashdreams.runtime_v2.user_input_events import UserInputEvents

    live_input = queue.SimpleQueue()
    loop: Any = Game2VUILoop(live_input=live_input)
    loop._initialize_loop_state()
    loop.presented_model_frames = lambda: ()
    type(loop).presented_model_frame_count = property(lambda self: 0)
    buffer = EventBuffer()
    buffer.register(0)
    buffer.register(1)
    buffer.append(UserInputEvents([_key("a", True)]))
    ui_events, _ = buffer.read(0)
    try:
        assert loop.step(0, ui_events) == []
    finally:
        del type(loop).presented_model_frame_count
    state = Game2VModelState.__new__(Game2VModelState)
    state.live_input = live_input
    assert _drain_live_game_events(state) == [{"action": "a", "phase": "down"}]
    assert _drain_live_game_events(state) == []
    # The model loop still sees the same key at its own step boundary.
    model_events, _ = buffer.read(1)
    assert [cast(Any, event).key for event in model_events.get_events()] == ["a"]


def test_sleep_until_frame_deadline_paces_evenly_and_skips_when_behind():
    sleeps: list[float] = []
    now = [0.0]

    def fake_now() -> float:
        return now[0]

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    _sleep_until_frame_deadline(
        started_at=0.0,
        frame_index=0,
        frame_count=4,
        duration_s=0.4,
        now=fake_now,
        sleep=fake_sleep,
    )
    assert sleeps == [pytest.approx(0.1)]
    now[0] = 0.25
    sleeps.clear()
    _sleep_until_frame_deadline(
        started_at=0.0,
        frame_index=1,
        frame_count=4,
        duration_s=0.4,
        now=fake_now,
        sleep=fake_sleep,
    )
    assert sleeps == []
    _sleep_until_frame_deadline(
        started_at=0.0,
        frame_index=2,
        frame_count=4,
        duration_s=0.4,
        now=fake_now,
        sleep=fake_sleep,
    )
    assert sleeps == [pytest.approx(0.05)]
    sleeps.clear()
    _sleep_until_frame_deadline(
        started_at=0.0,
        frame_index=0,
        frame_count=4,
        duration_s=0.0,
        now=fake_now,
        sleep=fake_sleep,
    )
    assert sleeps == []


def test_pace_duration_uses_last_generate_then_default():
    from types import SimpleNamespace

    from flashdreams.runtime_v2.session_desc import SessionDesc

    state = Game2VModelState.__new__(Game2VModelState)
    state.last_generate_wall_s = 0.43
    state.pipeline = SimpleNamespace(get_num_output_frames=lambda step: 12)
    state.session_desc = SessionDesc(frames_per_second_for_step=16)
    assert _pace_duration_s(state, 1) == pytest.approx(0.43)
    state.last_generate_wall_s = None
    assert _pace_duration_s(state, 1) == pytest.approx(0.4)


def test_file_renders_do_not_pace_the_game():
    """No live input in an MP4 run: a slow generate must not stretch the next render."""
    from flashdreams.runtime_v2.session_desc import PresentationMode, SessionDesc

    state = Game2VModelState.__new__(Game2VModelState)
    state.last_generate_wall_s = 1.37
    state.session_desc = SessionDesc(presentation_mode=PresentationMode.ON_DEMAND)
    assert _pace_duration_s(state, 1) == 0.0


def _stub_capture(chunk) -> list[dict]:
    """Mimic the capture page: an image and HUD snapshot on every model frame."""
    return [
        {
            "ui": _jpeg_data_url(np.full((2, 2, 3), 10 * (index + 1), dtype=np.uint8)),
            "state": {"moves": index},
        }
        for index, _packet in enumerate(chunk.frames)
    ]


def test_default_capture_size_matches_model_frame_size():
    from eden.apps.game2v.adapter import GAME2V_EDEN_WORLD_V2_14B_TAEHV_DEFAULTS

    assert (DEFAULT_CAPTURE_WIDTH, DEFAULT_CAPTURE_HEIGHT) == (
        GAME2V_EDEN_WORLD_V2_14B_TAEHV_DEFAULTS.pixel_width,
        GAME2V_EDEN_WORLD_V2_14B_TAEHV_DEFAULTS.pixel_height,
    )


class _StubRenderer:
    """Record the thread each browser call lands on and block until released."""

    instances: list["_StubRenderer"] = []

    def __init__(self, slug, *, width, height):
        self.slug = slug
        self.width = width
        self.height = height
        self.last_evaluate_s = 0.25
        self.last_decode_s = 0.05
        self.webgl_renderer = (
            "ANGLE (NVIDIA Corporation, NVIDIA L20Z/PCIe/SSE2, OpenGL ES 3.2)"
        )
        self.closed = False
        self.threads: list[str] = [threading.current_thread().name]
        self.release = threading.Event()
        self.started = threading.Event()
        self.renders = 0
        self.duration_s = None
        self.drain_calls = 0
        _StubRenderer.instances.append(self)

    def render_chunk(self, chunk, *, duration_s=0.0, drain_events=None):
        self.threads.append(threading.current_thread().name)
        self.duration_s = duration_s
        if drain_events is not None:
            for _ in chunk.frames:
                drain_events()
                self.drain_calls += 1
        self.started.set()
        self.release.wait(timeout=5)
        self.release.clear()
        self.started.clear()
        self.renders += 1
        return replace(chunk, frames=tuple(chunk.frames))

    def close(self):
        self.threads.append(threading.current_thread().name)
        self.closed = True


@pytest.fixture
def stub_renderer(monkeypatch):
    _StubRenderer.instances.clear()
    monkeypatch.setattr(game_renderer, "GameRenderer", _StubRenderer)
    yield _StubRenderer
    _StubRenderer.instances.clear()


def test_capture_worker_runs_on_its_own_thread_and_capture_waits(stub_renderer):
    timeline = GameTimeline("freeze")
    worker = GameCaptureWorker("freeze", width=832, height=480)
    try:
        pending = worker.submit(timeline.begin_chunk(12, 16))
        renderer = _wait_for_renderer(stub_renderer)
        # submit() returns immediately so Playwright stays off the model thread
        # and the session can generate the previous chunk while this one renders.
        assert not pending.done()
        renderer.release.set()
        chunk = pending.result(timeout=5)
        assert len(chunk.frames) == 12
        assert worker.last_evaluate_s == pytest.approx(0.25)
        assert worker.last_decode_s == pytest.approx(0.05)
        assert "NVIDIA" in worker.webgl_renderer
        assert renderer.width == 832 and renderer.height == 480
        worker_threads = {
            name for name in renderer.threads if name != threading.current_thread().name
        }
        assert worker_threads and all(
            name.startswith("game2v-render") for name in worker_threads
        )
        assert threading.current_thread().name not in renderer.threads

        blocking = threading.Event()
        captured = {}

        def _capture_next():
            captured["chunk"] = worker.capture(timeline.begin_chunk(12, 16))
            blocking.set()

        thread = threading.Thread(target=_capture_next)
        thread.start()
        if not renderer.started.wait(timeout=5):
            raise AssertionError("capture() never reached the browser thread")
        assert not blocking.is_set()
        renderer.release.set()
        thread.join(timeout=5)
        assert blocking.is_set()
        assert len(captured["chunk"].frames) == 12
        assert renderer.renders == 2
    finally:
        for instance in stub_renderer.instances:
            instance.release.set()
        worker.close()
    assert renderer.closed
    assert renderer.threads[-1].startswith("game2v-render")


def test_one_worker_overlaps_successive_chunks(stub_renderer):
    timeline = GameTimeline("freeze")
    worker = GameCaptureWorker("freeze")
    try:
        first = worker.submit(timeline.begin_chunk(12, 16))
        renderer = _wait_for_renderer(stub_renderer)
        second = worker.submit(timeline.begin_chunk(12, 16))
        # The executor is single-threaded, so the second chunk waits in the
        # queue; submit() itself must not block the model thread.
        assert not first.done()
        assert not second.done()
        renderer.release.set()
        first.result(timeout=5)
        if not renderer.started.wait(timeout=5):
            raise AssertionError("second chunk never reached the browser thread")
        renderer.release.set()
        second.result(timeout=5)
        assert renderer.renders == 2
        assert len(stub_renderer.instances) == 1
    finally:
        for instance in stub_renderer.instances:
            instance.release.set()
        worker.close()


def test_capture_worker_keeps_one_browser_across_chunks(stub_renderer):
    timeline = GameTimeline("freeze")
    worker = GameCaptureWorker("freeze")
    try:
        for _ in range(3):
            pending = worker.submit(timeline.begin_chunk(12, 16))
            renderer = _wait_for_renderer(stub_renderer)
            renderer.release.set()
            pending.result(timeout=5)
        assert len(stub_renderer.instances) == 1
    finally:
        for instance in stub_renderer.instances:
            instance.release.set()
        worker.close()


class _GpuOptions:
    """Call GameRenderer._gpu_launch_options without launching a browser."""

    def __init__(self) -> None:
        self._vendor_dir = None

    options = game_renderer.GameRenderer._gpu_launch_options

    def cleanup(self) -> None:
        if self._vendor_dir is not None:
            self._vendor_dir.cleanup()


def test_gpu_launch_options_register_nvidia_egl_and_keep_mesa_fallback(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("EDEN_GAME_GPU", raising=False)
    monkeypatch.delenv("EDEN_GAME_RASTER", raising=False)
    monkeypatch.setattr(game_renderer.sys, "platform", "linux")
    mesa = tmp_path / "50_mesa.json"
    mesa.write_text("{}")
    monkeypatch.setattr(game_renderer, "_MESA_EGL_VENDOR", mesa)
    holder = _GpuOptions()
    try:
        options = holder.options()
        assert options["args"] == list(game_renderer._GPU_LAUNCH_ARGS)
        assert holder._vendor_dir is not None
        vendor_path = Path(holder._vendor_dir.name) / "10_nvidia.json"
        # The variable replaces the vendor list, so Mesa must be named to remain
        # a fallback, and NVIDIA must come first to win when it works.
        assert (
            options["env"]["__EGL_VENDOR_LIBRARY_FILENAMES"] == f"{vendor_path}:{mesa}"
        )
        vendor = json.loads(vendor_path.read_text())
        assert vendor["ICD"]["library_path"] == "libEGL_nvidia.so.0"
    finally:
        holder.cleanup()


def test_gpu_launch_options_omit_mesa_when_absent(monkeypatch, tmp_path):
    monkeypatch.delenv("EDEN_GAME_GPU", raising=False)
    monkeypatch.delenv("EDEN_GAME_RASTER", raising=False)
    monkeypatch.setattr(game_renderer.sys, "platform", "linux")
    monkeypatch.setattr(game_renderer, "_MESA_EGL_VENDOR", tmp_path / "missing.json")
    holder = _GpuOptions()
    try:
        names = holder.options()["env"]["__EGL_VENDOR_LIBRARY_FILENAMES"]
        assert holder._vendor_dir is not None
        assert names == str(Path(holder._vendor_dir.name) / "10_nvidia.json")
        assert Path(names).is_file()
    finally:
        holder.cleanup()


@pytest.mark.parametrize(
    "raster, flag",
    [
        # WARP by default: the D3D11 readback otherwise waits on the GPU the
        # model saturates, which capped the live loop at ~11 fps on a 5090.
        (None, "--use-angle=d3d11-warp"),
        ("warp", "--use-angle=d3d11-warp"),
        ("gpu", "--use-angle=d3d11"),
        ("swiftshader", "--use-angle=swiftshader"),
    ],
)
def test_gpu_launch_options_on_windows(monkeypatch, raster, flag):
    """Windows has no EGL: no vendor JSON, no GLVND environment, ANGLE D3D11."""
    monkeypatch.delenv("EDEN_GAME_GPU", raising=False)
    if raster is None:
        monkeypatch.delenv("EDEN_GAME_RASTER", raising=False)
    else:
        monkeypatch.setenv("EDEN_GAME_RASTER", raster)
    monkeypatch.setattr(game_renderer.sys, "platform", "win32")
    holder = _GpuOptions()
    try:
        assert flag in holder.options()["args"]
        assert holder._vendor_dir is None
    finally:
        holder.cleanup()


@pytest.mark.parametrize("platform, raster", [("win32", "metal"), ("linux", "warp")])
def test_gpu_launch_options_reject_an_unusable_raster(monkeypatch, platform, raster):
    monkeypatch.delenv("EDEN_GAME_GPU", raising=False)
    monkeypatch.setenv("EDEN_GAME_RASTER", raster)
    monkeypatch.setattr(game_renderer.sys, "platform", platform)
    with pytest.raises(ValueError, match="EDEN_GAME_RASTER"):
        _GpuOptions().options()


def test_gpu_launch_options_disabled_by_env(monkeypatch):
    monkeypatch.setenv("EDEN_GAME_GPU", "0")
    holder = _GpuOptions()
    try:
        assert holder.options() == {"args": list(game_renderer._SOFTWARE_LAUNCH_ARGS)}
        assert holder._vendor_dir is None
    finally:
        holder.cleanup()


def _pipeline_state(monkeypatch, *, rank: str, frames_by_step: dict[int, int]):
    from types import SimpleNamespace

    from flashdreams.runtime_v2.session_desc import SessionDesc

    monkeypatch.setenv("RANK", rank)
    config = SimpleNamespace(
        game_slug="freeze",
        model_video_width=832,
        model_video_height=480,
        proxy_video=None,
        autopilot=None,
    )
    return Game2VModelState(
        pipeline=SimpleNamespace(
            get_num_output_frames=lambda step_index: frames_by_step[step_index],
        ),
        session_desc=SessionDesc(frames_per_second_for_step=16),
        config=cast(Any, config),
        keyboard_resampler=cast(
            Any,
            SimpleNamespace(
                input_timeline=SimpleNamespace(),
                keyboard_track=SimpleNamespace(),
            ),
        ),
    )


def test_begin_game_chunk_submits_on_rank_zero(stub_renderer, monkeypatch):
    state = _pipeline_state(monkeypatch, rank="0", frames_by_step={0: 9, 1: 12})
    try:
        pending = _begin_game_chunk(state, 0)
        renderer = _wait_for_renderer(stub_renderer)
        assert not pending.done()
        renderer.release.set()
        chunk = pending.result(timeout=5)
        assert len(chunk.frames) == 9
        assert state.game_capture is not None
    finally:
        for instance in stub_renderer.instances:
            instance.release.set()
        _discard_pending_game_chunk(state)
        if state.game_capture is not None:
            state.game_capture.close()


def test_begin_game_chunk_skips_the_browser_off_rank_zero(stub_renderer, monkeypatch):
    state = _pipeline_state(monkeypatch, rank="1", frames_by_step={0: 9})
    chunk = _begin_game_chunk(state, 0)
    assert chunk.frames[0].ui_frame is None
    assert len(chunk.frames) == 9
    assert state.game_capture is None
    assert not stub_renderer.instances


def test_collect_game_chunk_returns_a_frozen_packet_off_rank_zero(monkeypatch):
    state = _pipeline_state(monkeypatch, rank="1", frames_by_step={0: 9})
    state.pending_game_chunk = _begin_game_chunk(state, 0)
    collected = _collect_game_chunk(state)
    assert collected.frames[0].ui_frame is None
    assert state.pending_game_chunk is None
    with pytest.raises(RuntimeError, match="no pending game chunk"):
        _collect_game_chunk(state)


def test_step_pipeline_renders_the_successor_before_generate(
    stub_renderer, monkeypatch
):
    """The next chunk is submitted before generate, so it overlaps the GPU."""
    state = _pipeline_state(monkeypatch, rank="0", frames_by_step={0: 9, 1: 12})
    try:
        state.pending_game_chunk = _begin_game_chunk(state, 0, duration_s=0.0)
        renderer = _wait_for_renderer(stub_renderer)
        renderer.release.set()
        current = _collect_game_chunk(state)
        assert len(current.frames) == 9
        assert renderer.duration_s == 0.0
        state.last_generate_wall_s = 0.43
        successor = _begin_game_chunk(state, 1, duration_s=_pace_duration_s(state, 1))
        assert not successor.done()
        if not renderer.started.wait(timeout=5):
            raise AssertionError("successor chunk never reached the browser thread")
        renderer.release.set()
        next_chunk = successor.result(timeout=5)
        assert len(next_chunk.frames) == 12
        assert renderer.renders == 2
        assert renderer.duration_s == pytest.approx(0.43)
        assert renderer.drain_calls == 9 + 12
    finally:
        for instance in stub_renderer.instances:
            instance.release.set()
        _discard_pending_game_chunk(state)
        if state.game_capture is not None:
            state.game_capture.close()


def _wait_for_renderer(stub_renderer: type[_StubRenderer]) -> _StubRenderer:
    return _wait_for_renderers(stub_renderer, 1)[-1]


def _wait_for_renderers(
    stub_renderer: type[_StubRenderer], count: int
) -> list[_StubRenderer]:
    deadline = threading.Event()
    for _ in range(500):
        started = [
            instance
            for instance in stub_renderer.instances
            if instance.started.is_set()
        ]
        if len(started) >= count:
            return started
        deadline.wait(0.01)
    raise AssertionError(
        f"Expected {count} capture workers to start; got {len(stub_renderer.instances)}"
    )


def _jpeg_data_url(rgb: np.ndarray) -> str:
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
