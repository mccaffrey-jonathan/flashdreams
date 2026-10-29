# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Replay reader adapted for FlashDreams from ModelTC LightX2V (Apache-2.0),
# lightx2v/models/input_encoders/edenlive/recording.py at d2d7424.

"""Recorded proxy videos for Game2V: replaying one in place of the game, and writing one.

Replay reproduces the EdenLive reference reader (LightX2V's
``lightx2v/models/input_encoders/edenlive/recording.py`` ``RecordingReader``,
Apache-2.0) so a FlashDreams replay and the reference consume the same pixels:

* The length is ``4 * ((floor(duration * fps) - 1) // 4) + 1`` frames, capped by
  ``max_frames`` before the rounding.
* Frames are decoded once with PyAV as ``rgb24``. Target time ``k / fps`` takes
  the first decoded frame whose midpoint with its successor is at or after it,
  so ties go to the earlier frame and variable frame rates work.
* A source that is not already the model size is centre-cropped to the model
  aspect ratio, then resized with PIL bilinear.
* The model is conditioned on **global** video frames ``0, 4, 8, ...``, one per
  latent frame. That is not the live game's rule, which conditions on frames
  ``0, 4, 8`` of each chunk.

The reference decodes only the condition frames. This decodes every frame on
the same rule, because the frames between them are what a recording or a
side-by-side shows next to the generated video; the condition frames are the
same either way (``k / fps`` for ``k = 4 i`` is the reference's target).
"""

from __future__ import annotations

import json
import math
from collections.abc import Generator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from flashdreams.runtime_v2.video_encoder import Mp4Encoder

from .game_timeline import CONDITION_STRIDE, GameChunkPacket

_MIN_FRAMES = 5
"""Shortest replay the reference accepts: one chunk of two latent frames."""


def recording_frame_count(
    video_path: str | Path, *, fps: int, max_frames: int | None = None
) -> int:
    """Return the ``4n + 1`` frame count a replay of ``video_path`` generates.

    Args:
        video_path: Recorded proxy video.
        fps: Model frame rate the recording is resampled to.
        max_frames: Cap applied before rounding down to ``4n + 1``.

    Raises:
        ValueError: The container reports no duration, or fewer than five
            frames remain.
    """
    import av

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        if stream.duration is None or stream.time_base is None:
            raise ValueError(
                f"{video_path} reports no video stream duration; remux it to "
                "MP4 (ffmpeg -i in -c copy out.mp4)."
            )
        available = math.floor(float(stream.duration * stream.time_base) * fps)
    if max_frames is not None:
        available = min(available, max_frames)
    num_frames = 4 * ((available - 1) // 4) + 1
    if num_frames < _MIN_FRAMES:
        raise ValueError(
            f"{video_path} gives {num_frames} frames at {fps} fps; a replay "
            f"needs at least {_MIN_FRAMES}."
        )
    return num_frames


def iter_recording_frames(
    video_path: str | Path,
    *,
    height: int,
    width: int,
    fps: int,
    num_frames: int,
) -> Generator[np.ndarray, None, None]:
    """Yield ``num_frames`` RGB uint8 ``[H, W, 3]`` frames at ``k / fps``.

    The selection, crop and resize are the reference reader's, line for line,
    with every frame as a target instead of every fourth.
    """
    import av

    targets = np.arange(0, num_frames) / fps
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 16
        source_width, source_height = stream.width, stream.height
        crop_width = min(source_width, round(source_height * width / height))
        crop_height = min(source_height, round(source_width * height / width))
        left, top = (source_width - crop_width) // 2, (source_height - crop_height) // 2

        def rgb(frame: Any) -> np.ndarray:
            pixels = frame.to_ndarray(format="rgb24")
            if (source_width, source_height) != (width, height):
                image = Image.fromarray(pixels).crop(
                    (left, top, left + crop_width, top + crop_height)
                )
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
                # Midpoint ties choose the earlier frame, including VFR input.
                while target < len(targets) and targets[target] <= midpoint:
                    target += 1
                    yield pixels
                if target == len(targets):
                    return
            previous, previous_time = frame, frame_time
        pixels = rgb(previous)
        while target < len(targets):
            target += 1
            yield pixels


def is_replay_condition_frame(game_frame_id: int) -> bool:
    """Whether output frame ``game_frame_id`` (1-based) conditions the model in a replay."""
    return (game_frame_id - 1) % CONDITION_STRIDE == 0


class RecordingCaptureWorker:
    """Serve a recorded proxy video through the live capture worker's interface.

    The next chunk is decoded on a worker thread while the model generates the
    current one, as the live game's render is. Frames carry no HUD state, so a
    replay never pauses on a card or restarts a round.
    """

    def __init__(
        self,
        video_path: str | Path,
        *,
        width: int,
        height: int,
        fps: int,
        num_frames: int,
    ) -> None:
        self.video_path = Path(video_path)
        self.num_frames = num_frames
        self.last_evaluate_s = 0.0
        self.last_decode_s = 0.0
        self.webgl_renderer = ""
        self._frames = iter_recording_frames(
            video_path, height=height, width=width, fps=fps, num_frames=num_frames
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="game2v-replay"
        )

    def submit(
        self,
        chunk: GameChunkPacket,
        *,
        duration_s: float = 0.0,
        drain_events: Any = None,
    ) -> Future:
        """Queue the decode of ``chunk``'s frames and return immediately.

        ``duration_s`` and ``drain_events`` pace and feed a live game; a
        recording has nothing to pace or feed.
        """
        del duration_s, drain_events
        return self._executor.submit(self._read_chunk, chunk)

    def _read_chunk(self, chunk: GameChunkPacket) -> GameChunkPacket:
        import time

        started = time.perf_counter()
        frames = []
        for packet in chunk.frames:
            pixels = next(self._frames, None)
            if pixels is None:
                raise RuntimeError(
                    f"{self.video_path} ran out after {self.num_frames} frames; "
                    "the rollout asked for more chunks than the replay has."
                )
            condition = is_replay_condition_frame(packet.game_frame_id)
            frames.append(
                replace(
                    packet,
                    capture_condition=condition,
                    ui_frame=pixels,
                    condition_frame=pixels if condition else None,
                )
            )
        self.last_decode_s = time.perf_counter() - started
        return replace(chunk, frames=tuple(frames))

    def probe(self, *, drain_events: Any = None) -> dict:
        """A recording has no live page to poll."""
        del drain_events
        return {}

    def close(self) -> None:
        """Stop the decode thread and release the container."""
        try:
            self._executor.submit(self._frames.close).result()
        except RuntimeError:
            pass
        finally:
            self._executor.shutdown(wait=True)


def chunk_prompts(
    metadata: Mapping[str, Any],
    num_frames: int,
    *,
    fps: int,
    chunk_size: int,
    static_prompt: str | None = None,
) -> tuple[str, ...]:
    """Compose one prompt per AR chunk from an EdenLive ``prompts.json``.

    The rule of EdenLive's offline ``prompt_schedule``
    (``edenlive/data/recording_manifest.py``): chunk ``c`` covers latent frames
    ``[3c, 3c + 3)``, which decode to output frames ``[max(0, 4s - 3), 4e - 3)``;
    its prompt is ``staticPrompt`` followed by each distinct ``dynamicPrompts``
    text whose ``[startMs, endMs)`` overlaps those frames, in file order.

    Args:
        metadata: Parsed ``prompts.json`` with ``staticPrompt`` and optional
            ``dynamicPrompts`` (``startMs``, ``endMs``, ``text``).
        num_frames: Replay length in output frames (``4n + 1``).
        fps: Output frame rate.
        chunk_size: Latent frames per AR chunk.
        static_prompt: Replaces ``staticPrompt`` when given.
    """
    static = static_prompt if static_prompt else metadata.get("staticPrompt")
    if not isinstance(static, str) or not static.strip():
        raise ValueError(
            "prompts.json needs a non-empty staticPrompt (or pass --prompt)."
        )
    events = metadata.get("dynamicPrompts") or []
    if not isinstance(events, Sequence):
        raise ValueError("prompts.json dynamicPrompts must be a list.")
    latent_frames = (num_frames - 1) // 4 + 1
    prompts = []
    for start in range(0, latent_frames, chunk_size):
        end = min(start + chunk_size, latent_frames)
        start_ms = max(0, 4 * start - 3) * 1000 / fps
        end_ms = (4 * end - 3) * 1000 / fps
        texts = dict.fromkeys(
            event["text"]
            for event in events
            if event["startMs"] < end_ms and event["endMs"] > start_ms
        )
        prompts.append(" ".join([static, *texts]))
    return tuple(prompts)


def load_chunk_prompts(
    path: str | Path,
    num_frames: int,
    *,
    fps: int,
    chunk_size: int,
    static_prompt: str | None = None,
) -> tuple[str, ...]:
    """Read ``path`` and return :func:`chunk_prompts` for it."""
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{path} must hold a JSON object.")
    return chunk_prompts(
        metadata,
        num_frames,
        fps=fps,
        chunk_size=chunk_size,
        static_prompt=static_prompt,
    )


class ProxyRecorder:
    """Encode the proxy frames the model consumed to an MP4, off the model thread."""

    def __init__(self, path: str | Path, *, width: int, height: int, fps: int) -> None:
        self.path = Path(path)
        self._encoder = Mp4Encoder(
            path, width=width, height=height, frames_per_second=fps
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="game2v-proxy-record"
        )
        self._pending: Future | None = None

    def write(self, frames: Sequence[np.ndarray]) -> None:
        """Queue ``[H, W, 3]`` uint8 frames; raises an earlier write's failure."""
        if self._pending is not None and self._pending.done():
            self._pending.result()
        stacked = np.stack(frames)
        self._pending = self._executor.submit(self._encoder.write, stacked)

    def close(self) -> None:
        """Finish the file after every queued write."""
        try:
            self._executor.shutdown(wait=True)
            if self._pending is not None:
                self._pending.result()
        finally:
            self._encoder.close()


__all__ = [
    "ProxyRecorder",
    "RecordingCaptureWorker",
    "chunk_prompts",
    "is_replay_condition_frame",
    "iter_recording_frames",
    "load_chunk_prompts",
    "recording_frame_count",
]
