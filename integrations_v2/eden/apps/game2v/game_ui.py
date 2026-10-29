# SPDX-FileCopyrightText: Copyright (c) 2026 Vast AI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

# accessors; forwards game keys; optional proxy|generated side-by-side output;
# browser messages travel on an integration StepResult subclass.

"""Game2V UI loop: present the generated frame and hand game keys to the capture worker."""

from __future__ import annotations

import queue
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from flashdreams.api_v2.loop import IUILoop, ModelInferenceState
from flashdreams.runtime_v2.blit_model_output_to_screen_loop import _frame_to_layout
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.ui_compositing import prepare_ui_back_buffer
from flashdreams.runtime_v2.user_input_events import UserInputEvents

from .session import _game_input_payloads


@dataclass(frozen=True, slots=True, match_args=False)
class Game2VFrame(StepResult):
    """A presented frame plus the browser messages that describe it.

    Any window can present it as a plain :class:`StepResult`; the Game2V browser
    window also forwards :attr:`client_messages` on its control channel.
    """

    client_messages: tuple[Mapping[str, Any], ...] = ()
    """JSON payloads (``prompt``, ``game_ui``) for this frame, oldest first."""


class BrowserFeed:
    """Browser messages keyed by the presented frame they describe.

    The UI loop owns it as its state. The model loop fills it through
    ``invoke_async`` before it publishes the frames the messages describe, so
    they are in by the UI tick that presents those frames. Keys carry the reset
    generation: a reset never wipes messages already sent for the next one.
    """

    def __init__(self) -> None:
        self._pending: dict[tuple[int, int], list[Mapping[str, Any]]] = {}

    def add(
        self, generation: int, messages: Mapping[int, list[Mapping[str, Any]]]
    ) -> None:
        """Queue ``messages[n]`` for the ``n``-th frame presented in ``generation``."""
        for number, items in messages.items():
            self._pending.setdefault((generation, number), []).extend(items)

    def take(self, generation: int, presented: int) -> list[Mapping[str, Any]]:
        """Return the messages of every frame up to ``presented``, dropping stale ones."""
        due = sorted(
            key
            for key in self._pending
            if key[0] < generation or (key[0] == generation and key[1] <= presented)
        )
        taken: list[Mapping[str, Any]] = []
        for key in due:
            items = self._pending.pop(key)
            if key[0] == generation:
                taken.extend(items)
        return taken


class Game2VUILoop(IUILoop[Any]):
    """Emit each newly presented generated frame once.

    Channel 0 is the generated video and channel 1, when present, the proxy
    render it was conditioned on. A frame is emitted only when the presented
    frame changes, so an MP4 window writes every generated frame exactly once.
    """

    def __init__(
        self,
        *,
        live_input: queue.SimpleQueue | None = None,
        side_by_side: bool = False,
    ) -> None:
        """
        Args:
            live_input: Queue the capture worker drains between frames; every
                keyboard edge this loop reads is forwarded to it. The UI loop
                is the one loop that sees each input batch as it arrives, which
                is what lets keys land in the chunk being rendered rather than
                at the next model step.
            side_by_side: Present ``proxy | generated``, twice the model width.
        """
        self._live_input = live_input
        self._side_by_side = side_by_side

    def _initialize_loop_state(self) -> None:
        self._last_count = 0

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Forward game keys, then return the presented frame if it is new."""
        if self._live_input is not None and events is not None:
            for payload in _game_input_payloads(events):
                self._live_input.put(payload)
        frames = self.presented_model_frames()
        count = self.presented_model_frame_count
        if not frames or count == self._last_count:
            return []
        self._last_count = count
        frame = frames[0]
        if self._side_by_side:
            frame = side_by_side_frame(frames[1] if len(frames) > 1 else None, frame)
        output = _frame_to_layout(frame, self.output_layout)
        if isinstance(self.state, BrowserFeed):
            return [
                Game2VFrame(
                    step_index=step_index,
                    output=output,
                    frame_count=1,
                    output_layout=self.output_layout,
                    client_messages=tuple(self.state.take(self._generation, count)),
                )
            ]
        return [
            StepResult(
                step_index=step_index,
                output=output,
                frame_count=1,
                output_layout=self.output_layout,
            )
        ]

    def is_finished(self) -> bool:
        return (
            self.model_inference_state is ModelInferenceState.FINISHED
            and not self.has_pending_model_frames()
            and self._last_count == self.presented_model_frame_count
        )

    def reset(self) -> None:
        self._last_count = 0


def side_by_side_frame(
    proxy: torch.Tensor | None, generated: torch.Tensor
) -> torch.Tensor:
    """Place the proxy frame left of the generated one, in the generated frame's format.

    Args:
        proxy: ``[3, H', W']`` proxy render (uint8 on the CPU from the capture),
            or ``None`` for a black panel.
        generated: ``[3, H, W]`` generated frame in ``[-1, 1]``.

    Returns:
        ``[3, H, 2W]`` frame on the generated frame's device and dtype.
    """
    if proxy is None:
        left = torch.full_like(generated, -1.0)
    else:
        converted = prepare_ui_back_buffer(proxy, generated)
        assert converted is not None
        left = converted
    return torch.cat((left, generated), dim=-1)


__all__ = ["BrowserFeed", "Game2VFrame", "Game2VUILoop", "side_by_side_frame"]
