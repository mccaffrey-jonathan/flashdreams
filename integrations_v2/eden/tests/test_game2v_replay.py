# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recorded-proxy replay: frame selection, chunk layout, prompts and app sizing.

The reference is LightX2V's EdenLive ``RecordingReader``. Its selection rule is
re-implemented below from the published source; set
``EDEN_LIGHTX2V_RECORDING_PY`` to that file to also compare against it directly.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from eden.apps.game2v.game_timeline import GameTimeline
from eden.apps.game2v.recording import (
    ProxyRecorder,
    RecordingCaptureWorker,
    chunk_prompts,
    iter_recording_frames,
    recording_frame_count,
)
from PIL import Image

av = pytest.importorskip("av")

pytestmark = pytest.mark.ci_cpu


def _colour(index: int) -> tuple[int, int, int]:
    """A colour far from every other index's, so a lossy decode still identifies it."""
    return ((index * 37) % 256, (index // 7) * 36 % 256, 255 - (index * 11) % 256)


def _write_video(path: Path, *, width: int, height: int, fps: int, frames: int) -> None:
    """Solid frames whose colour encodes the source index (MPEG-4, always built in)."""
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = {"qscale": "1"}
        for index in range(frames):
            pixels = np.empty((height, width, 3), np.uint8)
            pixels[...] = _colour(index)
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(pixels, format="rgb24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _source_index(pixels: np.ndarray, frames: int) -> int:
    centre = pixels[pixels.shape[0] // 2, pixels.shape[1] // 2].astype(int)
    distances = [
        np.abs(centre - np.array(_colour(index))).sum() for index in range(frames)
    ]
    return int(np.argmin(distances))


def _reference_condition_frames(path, height, width, fps, max_frames=None):
    """LightX2V ``RecordingReader`` (frame count + ``_decode``), restated for the test."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        available = math.floor(float(stream.duration * stream.time_base) * fps)
    if max_frames is not None:
        available = min(available, max_frames)
    num_frames = 4 * ((available - 1) // 4) + 1
    targets = np.arange(0, num_frames, 4) / fps
    out = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        sw, sh = stream.width, stream.height
        cw = min(sw, round(sh * width / height))
        ch = min(sh, round(sw * height / width))
        left, top = (sw - cw) // 2, (sh - ch) // 2

        def rgb(frame):
            pixels = frame.to_ndarray(format="rgb24")
            if (sw, sh) != (width, height):
                image = Image.fromarray(pixels).crop((left, top, left + cw, top + ch))
                pixels = np.asarray(
                    image.resize((width, height), Image.Resampling.BILINEAR)
                )
            return pixels

        decoded = container.decode(video=0)
        previous = next(decoded)
        origin, previous_time, target = float(previous.time), 0.0, 0
        for frame in decoded:
            frame_time = float(frame.time) - origin
            midpoint = (previous_time + frame_time) / 2
            if targets[target] <= midpoint:
                pixels = rgb(previous)
                while target < len(targets) and targets[target] <= midpoint:
                    target += 1
                    out.append(pixels)
                if target == len(targets):
                    return num_frames, out
            previous, previous_time = frame, frame_time
        pixels = rgb(previous)
        while target < len(targets):
            target += 1
            out.append(pixels)
    return num_frames, out


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """3.5 s at 30 fps and 640x360: off-rate and off-size, so resampling and the crop run."""
    path = tmp_path_factory.mktemp("replay") / "proxy.mp4"
    _write_video(path, width=640, height=360, fps=30, frames=105)
    return path


def test_frame_count_is_4n_plus_1_of_the_duration(video):
    assert (
        recording_frame_count(video, fps=24)
        == 4 * ((math.floor(3.5 * 24) - 1) // 4) + 1
        == 81
    )
    assert recording_frame_count(video, fps=24, max_frames=40) == 37
    with pytest.raises(ValueError, match="at least 5"):
        recording_frame_count(video, fps=24, max_frames=4)


def test_condition_frames_match_the_reference_reader_exactly(video):
    num_frames, reference = _reference_condition_frames(video, 72, 128, 24)
    ours = list(
        iter_recording_frames(
            video, height=72, width=128, fps=24, num_frames=num_frames
        )
    )
    assert len(ours) == num_frames
    assert len(reference) == (num_frames - 1) // 4 + 1
    for latent, expected in enumerate(reference):
        assert np.array_equal(ours[4 * latent], expected), latent


def test_every_frame_follows_the_midpoint_rule(video):
    """Frame ``k`` shows the source frame nearest ``k / 24`` s; ties take the earlier one."""
    frames = list(
        iter_recording_frames(video, height=72, width=128, fps=24, num_frames=81)
    )
    assert frames[0].shape == (72, 128, 3)
    for k, pixels in enumerate(frames):
        exact = Fraction(k * 30, 24) - Fraction(1, 2)
        expected = min(math.ceil(exact), 104)
        allowed = {expected, expected + 1} if exact.denominator == 1 else {expected}
        assert _source_index(pixels, 105) in allowed, k


def test_condition_frames_land_on_global_multiples_of_four(video):
    """Chunk 0 conditions on 0,4,8; chunk c >= 1 on 12c, 12c+4, 12c+8 (not 9,13,17)."""
    worker = RecordingCaptureWorker(video, width=128, height=72, fps=24, num_frames=81)
    timeline = GameTimeline("")
    try:
        all_frames = list(
            iter_recording_frames(video, height=72, width=128, fps=24, num_frames=81)
        )
        seen = 0
        for count in (9, 12, 12):
            chunk = worker.submit(timeline.begin_chunk(count, 24)).result(timeout=30)
            conditioned = [
                seen + index
                for index, frame in enumerate(chunk.frames)
                if frame.condition_frame is not None
            ]
            assert conditioned == [g for g in range(seen, seen + count) if g % 4 == 0]
            assert len(conditioned) == 3
            for index, frame in enumerate(chunk.frames):
                assert np.array_equal(frame.ui_frame, all_frames[seen + index])
                assert frame.ui_state is None
            seen += count
        assert seen == 33
    finally:
        worker.close()


def test_replay_worker_refuses_to_run_past_the_video(video):
    worker = RecordingCaptureWorker(video, width=128, height=72, fps=24, num_frames=9)
    try:
        timeline = GameTimeline("")
        worker.submit(timeline.begin_chunk(9, 24)).result(timeout=30)
        with pytest.raises(RuntimeError, match="ran out"):
            worker.submit(timeline.begin_chunk(12, 24)).result(timeout=30)
    finally:
        worker.close()


@pytest.mark.skipif(
    not os.environ.get("EDEN_LIGHTX2V_RECORDING_PY"),
    reason="set EDEN_LIGHTX2V_RECORDING_PY to LightX2V's edenlive/recording.py",
)
def test_matches_lightx2v_recording_reader_file(video):
    spec = importlib.util.spec_from_file_location(
        "lightx2v_recording", os.environ["EDEN_LIGHTX2V_RECORDING_PY"]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with module.RecordingReader(str(video), 72, 128, 24, None) as reader:
        reference = reader.next_chunk().permute(1, 2, 3, 0).numpy()
        num_frames = reader.num_frames
    ours = list(
        iter_recording_frames(
            video, height=72, width=128, fps=24, num_frames=num_frames
        )
    )
    assert np.array_equal(np.stack(ours[::4]), reference)


def test_chunk_prompts_follow_the_offline_schedule():
    metadata = {
        "staticPrompt": "A lava cave.",
        "dynamicPrompts": [
            {"startMs": 0, "endMs": 100, "text": "A flash."},
            {"startMs": 2400, "endMs": 2600, "text": "A jump."},
            {"startMs": 2500, "endMs": 9000, "text": "A jump."},
        ],
    }
    # 117 frames = 30 latents = 10 chunks; chunk c covers frames [max(0,12c-3), 12c+9).
    prompts = chunk_prompts(metadata, 117, fps=24, chunk_size=3)
    assert len(prompts) == 10
    assert prompts[0] == "A lava cave. A flash."
    assert prompts[1] == "A lava cave."
    # 2400 ms = frame 57.6: chunk 4 is frames [45, 57) and ends before it.
    assert prompts[4] == "A lava cave."
    assert prompts[5] == "A lava cave. A jump."
    assert prompts[9] == "A lava cave. A jump."
    assert (
        chunk_prompts(metadata, 117, fps=24, chunk_size=3, static_prompt="X")[1] == "X"
    )
    with pytest.raises(ValueError, match="staticPrompt"):
        chunk_prompts({"dynamicPrompts": []}, 117, fps=24, chunk_size=3)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg on PATH")
def test_proxy_recorder_writes_every_frame(tmp_path):
    path = tmp_path / "proxy.mp4"
    recorder = ProxyRecorder(path, width=64, height=32, fps=24)
    frames = [np.full((32, 64, 3), 10 * index, np.uint8) for index in range(21)]
    recorder.write(frames[:9])
    recorder.write(frames[9:])
    recorder.close()
    with av.open(str(path)) as container:
        assert sum(1 for _ in container.decode(video=0)) == 21


def _replay_app():
    from eden.apps.game2v.adapter import create_app_v2_5b_taehv

    return create_app_v2_5b_taehv()


def test_replay_sizes_the_rollout_and_noise_to_the_whole_video(tmp_path):
    path = tmp_path / "proxy.mp4"
    _write_video(
        path, width=128, height=72, fps=24, frames=125
    )  # 125 -> 121 frames, T=31
    cover = tmp_path / "cover.png"
    Image.new("RGB", (64, 36)).save(cover)
    app = _replay_app()
    base = [
        "--device",
        "cpu",
        "--proxy-video",
        str(path),
        "--first-frame",
        str(cover),
        "--prompt",
        "p",
    ]
    # 31 latent frames is not whole 3-latent chunks: the error names the cap that is.
    with pytest.raises(ValueError, match=r"--max-frames 117"):
        app.init(base)
    app.init([*base, "--max-frames", "117"])
    assert app._proxy_frames == 117
    assert app._total_blocks == 10
    assert app.pipeline_config.diffusion_model.max_latent_frames == 30
    assert app._input_values["proxy_video"] == path
    # A shorter rollout is a prefix of the same noise; a longer one is refused.
    app.init([*base, "--max-frames", "117", "--total-blocks", "4"])
    assert app._total_blocks == 4
    with pytest.raises(ValueError, match=r"--total-blocks must be <= 10"):
        app.init([*base, "--max-frames", "117", "--total-blocks", "11"])


def test_replay_prompts_json_turns_on_per_chunk_encoding_only_when_text_changes(
    tmp_path,
):
    path = tmp_path / "proxy.mp4"
    _write_video(path, width=128, height=72, fps=24, frames=120)
    cover = tmp_path / "cover.png"
    Image.new("RGB", (64, 36)).save(cover)
    prompts = tmp_path / "prompts.json"
    base = ["--device", "cpu", "--proxy-video", str(path), "--first-frame", str(cover)]
    prompts.write_text(json.dumps({"staticPrompt": "S.", "dynamicPrompts": []}))
    app = _replay_app()
    app.init([*base, "--prompts-json", str(prompts)])
    assert app._prompt_schedule == ("S.",) * 10
    assert app._input_values["prompt"] == "S."
    assert app._dynamic_prompt is False
    prompts.write_text(
        json.dumps(
            {
                "staticPrompt": "S.",
                "dynamicPrompts": [{"startMs": 3000, "endMs": 4000, "text": "E."}],
            }
        )
    )
    app.init([*base, "--prompts-json", str(prompts)])
    assert app._dynamic_prompt is True
    assert app._prompt_schedule[-1] == "S."
    assert "S. E." in app._prompt_schedule


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--dynamic-prompt"], "--dynamic-prompt does not apply"),
        (["--wait-for-start"], "--wait-for-start does not apply"),
        (["--autopilot"], "--autopilot does not apply"),
        (["--max-frames", "3"], "--max-frames must be >= 9"),
    ],
)
def test_replay_rejects_live_only_flags(tmp_path, extra, message):
    path = tmp_path / "proxy.mp4"
    _write_video(path, width=128, height=72, fps=24, frames=30)
    with pytest.raises(ValueError, match=message):
        _replay_app().init(["--device", "cpu", "--proxy-video", str(path), *extra])


def test_replay_resolver_needs_a_first_frame_and_prompt(tmp_path):
    from eden.apps.game2v.game14b import resolve_game14b_conditioning

    cover = tmp_path / "cover.png"
    Image.new("RGB", (64, 36)).save(cover)
    with pytest.raises(ValueError, match="--first-frame and --prompt"):
        resolve_game14b_conditioning({"proxy_video": "x.mp4", "prompt": "p"})
    conditioning = resolve_game14b_conditioning(
        {"proxy_video": "x.mp4", "image_path": cover, "prompt": "p"}
    )
    assert conditioning.first_frame_path == cover
    assert conditioning.prompt == "p"


def test_replay_prompt_follows_the_schedule_by_chunk():
    from eden.apps.game2v.session import _compose_chunk_prompt

    encoded = []
    state: Any = SimpleNamespace(
        config=SimpleNamespace(
            proxy_video=Path("x.mp4"),
            prompt_schedule=("A.", "A.", "A. B."),
            conditioning=SimpleNamespace(prompt="A."),
            dynamic_prompt=True,
        ),
        static_prompt="",
        active_prompt="",
        active_event_lines=(),
        encoded_prompt="",
        blocks_generated=0,
        ar_index=0,
        prompt_encode_s=0.0,
        cache=object(),
        pipeline=SimpleNamespace(
            encode_text=lambda text: encoded.append(text) or text,
            replace_text_embeddings=lambda cache, embeddings: None,
        ),
    )
    assert _compose_chunk_prompt(state) == ("A.", ())
    assert encoded == []
    for index in (1, 2):
        state.ar_index = state.blocks_generated = index
        _compose_chunk_prompt(state)
    assert state.active_prompt == "A. B."
    assert encoded == [["A. B."]]
