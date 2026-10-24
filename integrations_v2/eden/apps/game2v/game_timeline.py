# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic 60 Hz chunk/frame metadata for Game2V.

This timeline never runs the authoritative simulation and does not freeze live
keys. Tick targets are computed here; Chromium consumes keyboard/UI events as
they arrive while the capture worker paces the in-flight chunk.
"""

from __future__ import annotations

from collections import deque
from fractions import Fraction
import math
from dataclasses import dataclass
from typing import Literal

GAME_SIM_HZ = 60
GAME_SIM_DT = 1.0 / GAME_SIM_HZ
CONDITION_STRIDE = 4
InputPhase = Literal["down", "up"]


@dataclass(frozen=True, slots=True)
class GameInputEvent:
    sequence: int
    action: str
    phase: InputPhase


@dataclass(frozen=True, slots=True)
class GameInputSnapshot:
    held_actions: frozenset[str]
    edge_actions: tuple[str, ...]
    sequence: int


@dataclass(frozen=True, slots=True)
class GameFramePacket:
    game_frame_id: int
    logical_time_s: float
    simulation_tick: int
    input_sequence: int
    game_slug: str
    capture_condition: bool = False
    ui_frame: object | None = None
    condition_frame: object | None = None
    ui_state: object | None = None
    """JSON-serializable HUD snapshot for this frame, or None.

    Produced by the game's own transport module. The runtime forwards it
    without interpreting the contents.
    """


@dataclass(frozen=True, slots=True)
class GameChunkPacket:
    chunk_id: int
    start_time_s: float
    end_time_s: float
    input_sequence_start: int
    input_sequence_end: int
    simulation_tick_start: int
    simulation_tick_end: int
    frames: tuple[GameFramePacket, ...]
    input_events: tuple[GameInputEvent, ...] = ()


class GameInputRouter:
    """Maintain held state and ordered edge pulses independently."""

    def __init__(self) -> None:
        self._held: set[str] = set()
        self._edges: deque[GameInputEvent] = deque()
        self._next_sequence = 0
        self.events: list[GameInputEvent] = []

    def push(self, action: str, phase: InputPhase) -> GameInputEvent:
        if not action:
            raise ValueError("action must be non-empty")
        if phase not in ("down", "up"):
            raise ValueError(f"unsupported input phase: {phase!r}")
        self._next_sequence += 1
        event = GameInputEvent(self._next_sequence, action, phase)
        self.events.append(event)
        if phase == "down":
            # Repeated keydown events are held-state noise. A new down after an
            # up remains a distinct edge pulse and preserves event ordering.
            if action not in self._held:
                self._edges.append(event)
            self._held.add(action)
        else:
            self._held.discard(action)
        return event

    def release_all(self) -> None:
        self._held.clear()

    def snapshot_for_chunk(self) -> GameInputSnapshot:
        edges = tuple(event.action for event in self._edges)
        self._edges.clear()
        return GameInputSnapshot(
            held_actions=frozenset(self._held),
            edge_actions=edges,
            sequence=self._next_sequence,
        )

    @property
    def held_actions(self) -> frozenset[str]:
        return frozenset(self._held)


class GameTimeline:
    """Create deterministic 60 Hz chunk/frame metadata without wall-clock waits."""

    def __init__(
        self,
        game_slug: str,
        *,
        simulation_hz: int = GAME_SIM_HZ,
        condition_stride: int = CONDITION_STRIDE,
    ) -> None:
        if simulation_hz <= 0:
            raise ValueError("simulation_hz must be positive")
        if isinstance(condition_stride, bool) or not isinstance(condition_stride, int) or condition_stride <= 0:
            raise ValueError("condition_stride must be a positive integer")
        self.game_slug = game_slug
        self.simulation_hz = simulation_hz
        self.condition_stride = condition_stride
        self.input = GameInputRouter()
        self.logical_time_s = 0.0
        self.simulation_tick = 0
        self.chunk_id = 0
        self.game_frame_id = 0
        self._time = Fraction(0)
        self._last_sequence = 0

    def begin_chunk(self, frame_count: int, frames_per_second: float) -> GameChunkPacket:
        """Space ``frame_count`` captures across ``frame_count / fps`` of 60 Hz sim.

        Simulation stays at ``simulation_hz``. The model fps only chooses which
        ticks become captured frames, so a 16 fps model samples every ~3.75
        ticks and a 24 fps model every 2.5, with no other rate baked in.
        """
        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if not math.isfinite(frames_per_second) or frames_per_second <= 0:
            raise ValueError("frames_per_second must be positive")
        snapshot = self.input.snapshot_for_chunk()
        start_tick = self.simulation_tick
        start_time = self.logical_time_s
        tick_count = round((self._time + Fraction(frame_count) / Fraction(str(frames_per_second))) * self.simulation_hz) - start_tick
        if tick_count <= 0:
            raise ValueError("chunk duration must contain at least one simulation tick")
        # One condition frame per latent frame, at a fixed position inside the
        # chunk. The model's chunk is ``len_t`` latent frames and the stride is
        # the VAE's temporal ratio, so this lands on exactly ``len_t`` frames
        # (0, 4, 8, ...) regardless of how many chunks came before. Keying it on
        # the running frame count instead drifts by one frame every chunk,
        # since a causal first chunk is 9 frames and the rest are 12 — neither
        # divisible by the stride — and a restart then opens its new first
        # chunk mid-phase and comes up a condition frame short.
        frames: list[GameFramePacket] = []
        for frame_index in range(frame_count):
            tick = round((self._time + Fraction(frame_index + 1) / Fraction(str(frames_per_second))) * self.simulation_hz)
            capture_condition = (frame_index % self.condition_stride) == 0
            self.game_frame_id += 1
            frames.append(GameFramePacket(
                game_frame_id=self.game_frame_id,
                logical_time_s=start_time + (frame_index + 1) / frames_per_second,
                simulation_tick=tick,
                input_sequence=snapshot.sequence,
                game_slug=self.game_slug,
                capture_condition=capture_condition,
            ))
        self.simulation_tick += tick_count
        self._time += Fraction(frame_count) / Fraction(str(frames_per_second))
        self.logical_time_s = float(self._time)
        self.chunk_id += 1
        input_events = tuple(self.input.events)
        self.input.events.clear()
        previous_sequence = self._last_sequence
        self._last_sequence = snapshot.sequence
        return GameChunkPacket(
            chunk_id=self.chunk_id,
            start_time_s=start_time,
            end_time_s=self.logical_time_s,
            input_sequence_start=previous_sequence,
            input_sequence_end=snapshot.sequence,
            simulation_tick_start=start_tick,
            simulation_tick_end=self.simulation_tick,
            frames=tuple(frames),
            input_events=input_events,
        )

    def reset(self) -> None:
        self.input = GameInputRouter()
        self.logical_time_s = 0.0
        self.simulation_tick = 0
        self.chunk_id = 0
        self.game_frame_id = 0
        self._time = Fraction(0)
        self._last_sequence = 0
