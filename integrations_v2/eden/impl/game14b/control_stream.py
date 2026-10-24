# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Overlapped proxy-DINO encoding of rendered game frames.

The point of this module is latency hiding. A chunk's DINO features depend
only on that chunk's own rendered frames, so the encode can start the instant
the render lands and be awaited only when that chunk is about to denoise.

The timing follows the session's existing pipelining. Chunk ``N+1``'s render
is submitted *before* chunk ``N``'s ``generate``, so it completes while the
DiT is denoising chunk ``N``. Chaining the encode onto the render future
(rather than onto the later ``_collect_game_chunk``) is what puts chunk
``N+1``'s DINO cost inside chunk ``N``'s denoising; by the time ``take(N+1)``
runs a step later, the features are already there. Chunk 0 is submitted and
consumed within the same step, so its encode is unavoidably exposed.

Two mechanisms make the overlap real rather than nominal:

* A dedicated CUDA stream. Work queued on the default stream would simply
  serialize behind the DiT kernels; the side stream lets the encode occupy
  gaps instead, and a recorded event orders the consumer after it.
* A single-worker thread, so the Python-side submission order matches step
  order and only one encode is ever in flight.

Rank topology: only rank 0 renders, so only rank 0 can encode. Features are
broadcast to the other ranks, which is far cheaper than replicating a ViT-L
forward — about 15 MiB of bf16 per chunk at 480x832 with ``len_t=3``.
Sequence parallelism is deliberately not used: the DINO sequence is 576 patch
tokens, short enough that the collectives would cost more than the sharded
compute saves.
"""

from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np
import torch

from .dino import ProxyDinoControlEncoder, ProxyDinoEncoderConfig


def _is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _world_size() -> int:
    return torch.distributed.get_world_size() if _is_distributed() else 1


def _rank() -> int:
    if _is_distributed():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))


class _PendingEncode:
    """One in-flight encode plus the event that orders its consumer."""

    __slots__ = ("future", "event")

    def __init__(self, future: Future) -> None:
        self.future = future
        self.event: torch.cuda.Event | None = None


class ProxyDinoControlStream:
    """Encode each chunk's condition frames a step ahead of their use.

    Not thread-safe by design: the model loop is the only caller and its
    ``submit`` / ``take`` calls are already serialized by step order.
    """

    def __init__(
        self,
        config: ProxyDinoEncoderConfig,
        *,
        latent_height: int,
        latent_width: int,
        len_t: int,
        max_latent_frames: int | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.config = config
        self.latent_height = int(latent_height)
        self.latent_width = int(latent_width)
        self.len_t = int(len_t)
        self.max_latent_frames = max_latent_frames
        self.dtype = dtype
        self._encoder: ProxyDinoControlEncoder | None = None
        self._pending: dict[int, _PendingEncode] = {}
        self._is_source = _rank() == 0
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="game2v-dino")
            if self._is_source
            else None
        )
        # Side stream so the encode can fill gaps in the DiT's occupancy
        # instead of queueing behind it.
        self._stream = (
            torch.cuda.Stream()
            if self._is_source and torch.cuda.is_available()
            else None
        )

    @property
    def control_dim(self) -> int:
        return self._ensure_encoder().feature_dim

    def _ensure_encoder(self) -> ProxyDinoControlEncoder:
        if self._encoder is None:
            self._encoder = self.config.setup()
        return self._encoder

    def prepare(self) -> None:
        """Build the encoder eagerly, before the first chunk needs it."""
        if self._is_source:
            self._ensure_encoder()

    def submit(self, autoregressive_index: int, pending_render: Any) -> None:
        """Begin encoding as soon as ``pending_render`` produces its frames.

        Args:
            autoregressive_index: AR step these frames condition.
            pending_render: The in-flight render — a ``Future`` yielding a
                ``GameChunkPacket`` on rank zero, or the packet itself.

        Chaining on the future is the whole trick: the encode starts when the
        render lands (mid-denoise of the previous chunk), not when the session
        later collects it. Ranks that do not render return immediately and
        pick the features up from :meth:`take`'s broadcast.
        """
        if self._executor is None:
            return
        if autoregressive_index in self._pending:
            raise RuntimeError(
                f"a proxy-DINO encode is already queued for AR step "
                f"{autoregressive_index}"
            )
        entry = _PendingEncode(
            self._executor.submit(
                self._await_and_encode, pending_render, autoregressive_index
            )
        )
        self._pending[autoregressive_index] = entry

    def _await_and_encode(
        self, pending_render: Any, autoregressive_index: int
    ) -> tuple[torch.Tensor, Any]:
        chunk = (
            pending_render.result()
            if isinstance(pending_render, Future)
            else pending_render
        )
        frames = self.len_t
        if self.max_latent_frames is not None:
            frames = min(
                frames, self.max_latent_frames - autoregressive_index * self.len_t
            )
        images = _stack_condition_frames(chunk.frames, expected=frames)
        encoder = self._ensure_encoder()
        if self._stream is None:
            return (
                encoder.encode(
                    images,
                    latent_height=self.latent_height,
                    latent_width=self.latent_width,
                    dtype=self.dtype,
                ),
                None,
            )
        with torch.cuda.stream(self._stream):
            control = encoder.encode(
                images,
                latent_height=self.latent_height,
                latent_width=self.latent_width,
                dtype=self.dtype,
            )
            event = torch.cuda.Event()
            event.record(self._stream)
        return control, event

    def take(self, autoregressive_index: int) -> torch.Tensor:
        """Return the features for ``autoregressive_index``, waiting if needed.

        In steady state the encode was submitted a step earlier and has long
        finished, so this is a handoff rather than a wait.
        """
        control = self._take_local(autoregressive_index)
        return self._broadcast(control, autoregressive_index)

    def _take_local(self, autoregressive_index: int) -> torch.Tensor | None:
        if self._executor is None:
            return None
        entry = self._pending.pop(autoregressive_index, None)
        if entry is None:
            raise RuntimeError(
                f"no proxy-DINO encode was submitted for AR step "
                f"{autoregressive_index}; submit() must run when that chunk's "
                "render is started."
            )
        control, event = entry.future.result()
        if event is not None:
            # Order the consuming (default) stream after the side stream's
            # encode without a host-side device synchronize.
            torch.cuda.current_stream().wait_event(event)
            # The tensor was allocated by the side stream; tell the caching
            # allocator the default stream now reads it.
            control.record_stream(torch.cuda.current_stream())
        return control

    def _broadcast(
        self, control: torch.Tensor | None, autoregressive_index: int
    ) -> torch.Tensor:
        """Share rank zero's features with every rank."""
        if _world_size() == 1:
            assert control is not None, "single-rank stream must own its encode"
            return control
        if control is None:
            frames = self.len_t
            if self.max_latent_frames is not None:
                frames = min(
                    frames, self.max_latent_frames - autoregressive_index * self.len_t
                )
            control = torch.empty(
                (frames, self.control_dim, self.latent_height, self.latent_width),
                device=self.config.device,
                dtype=self.dtype,
            )
        torch.distributed.broadcast(control, src=0)
        return control

    def reset(self) -> None:
        """Drop queued encodes; used when a restart rewinds the rollout."""
        for entry in self._pending.values():
            if not entry.future.cancel():
                try:
                    entry.future.result()
                except Exception:
                    pass
        self._pending.clear()

    def close(self) -> None:
        self.reset()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


def _stack_condition_frames(frames: Any, *, expected: int) -> torch.Tensor:
    """Collect a chunk's condition renders into ``[T, 3, H, W]`` in ``[0, 1]``.

    The renderer tags every ``CONDITION_STRIDE``-th frame with a
    ``condition_frame``, which is one per latent frame, so the count must
    equal the chunk's latent length.
    """
    renders = [
        frame.condition_frame
        for frame in frames
        if getattr(frame, "condition_frame", None) is not None
    ]
    if len(renders) != expected:
        raise RuntimeError(
            f"chunk carries {len(renders)} condition frames but the model "
            f"consumes {expected}; the capture stride and len_t disagree."
        )
    stacked = np.stack(renders)  # [T, H, W, 3] uint8
    tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()
    if torch.cuda.is_available():
        tensor = tensor.pin_memory()
    return tensor.float().div_(255.0)


__all__ = ["ProxyDinoControlStream"]
