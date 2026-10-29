# SPDX-FileCopyrightText: Copyright (c) 2026 Vast AI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for FlashDreams v0.2.0: the EdenLive fork patched
# flashdreams/infra/diffusion/model/base.py in place; the same behaviour lives
# here as a subclass so core numerics stay unchanged.

"""Diffusion model with the EdenLive reference noise layout.

:class:`EdenDiffusionModel` adds two knobs to :class:`DiffusionModel`:

``noise_from_rank0``
    Every noise tensor -- the initial chunk noise and each scheduler re-noise --
    is drawn at the full *unpatchified* ``(..., C, T, H, W)`` shape, as the
    reference implementations do, then moved to FlashDreams' ``(..., T, C, H,
    W)`` layout and patchified. Under context parallelism rank 0 draws and
    broadcasts, so each shard receives a disjoint slice of one sample rather
    than the same noise repeated per rank.

``max_latent_frames``
    For bounded replays (a recorded proxy of known length): draw the *whole
    video's* initial noise once, ``(..., C, max_latent_frames, H, W)``, park it
    on the CPU and slice ``len_t`` frames per AR step. With
    ``noise_from_rank0`` this reproduces the RNG stream of LightX2V's EdenLive
    runner (one whole-video draw, then three re-noise draws per chunk), so a
    seed gives the same noise on both sides. Only whole chunks are supported:
    ``max_latent_frames`` must be a multiple of ``len_t``.

It also stages the cache-commit timestep in the scheduler's network timestep
dtype (see :class:`~eden.impl.scheduler.EdenFlowMatchScheduler`), so the denoise
and commit forwards share one captured CUDA graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from flashdreams.infra.diffusion.model import DiffusionModel, DiffusionModelConfig


@dataclass(kw_only=True)
class EdenDiffusionModelConfig(DiffusionModelConfig):
    """:class:`DiffusionModelConfig` plus the EdenLive noise knobs."""

    _target: type["EdenDiffusionModel"] = field(
        default_factory=lambda: EdenDiffusionModel
    )

    noise_from_rank0: bool = False
    """Draw all noise at the unpatchified ``(C, T, H, W)`` shape (on rank 0
    under CP), then patchify. Needs ``context_noise == 0``."""

    max_latent_frames: int | None = None
    """Bounded replay length in latent frames: draw the whole video's initial
    noise once and slice it per chunk. Must be a multiple of ``len_t``."""


class EdenDiffusionModel(DiffusionModel):
    """:class:`DiffusionModel` with the EdenLive reference noise layout."""

    config: EdenDiffusionModelConfig

    def __init__(self, config: EdenDiffusionModelConfig) -> None:
        # Validate before the transformer is built: that loads the weights.
        if config.noise_from_rank0 and config.noise_in_unpatchified_shape:
            raise ValueError(
                "noise_from_rank0 already draws in the unpatchified shape; "
                "do not also set noise_in_unpatchified_shape"
            )
        if config.noise_from_rank0 and config.context_noise > 0:
            # ``finalize``'s ``add_noise`` would still draw per rank.
            raise ValueError("noise_from_rank0 requires context_noise == 0")
        if config.noise_in_unpatchified_shape:
            raise ValueError(
                "EdenDiffusionModel draws unpatchified noise through "
                "noise_from_rank0; noise_in_unpatchified_shape is not supported"
            )
        if config.max_latent_frames is not None and config.max_latent_frames <= 0:
            raise ValueError("max_latent_frames must be positive")
        super().__init__(config)
        self.config = config
        len_t = getattr(self.transformer.config, "len_t", None)
        if (
            config.max_latent_frames is not None
            and len_t is not None
            and config.max_latent_frames % len_t
        ):
            raise ValueError(
                f"max_latent_frames={config.max_latent_frames} must be a multiple "
                f"of len_t={len_t}; partial final chunks are not supported"
            )
        self._len_t: int = int(len_t) if len_t is not None else 1
        self._initial_video_noise: Tensor | None = None

    def reset_noise(self) -> None:
        """Rewind the RNG and drop the cached whole-video noise.

        Call between rollouts so a new session with the same seed reproduces
        the same noise.
        """
        self._rng = None
        self._initial_video_noise = None

    @staticmethod
    def _distributed() -> bool:
        return (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )

    def _full_shape(self, like: Tensor) -> tuple[int, ...]:
        """The unpatchified ``(..., T, C, H, W)`` shape behind patchified ``like``."""
        return tuple(
            self.transformer.unpatchify_and_maybe_gather_cp(
                torch.empty_like(like)
            ).shape
        )

    def _broadcast_and_shard(self, full: Tensor) -> Tensor:
        if self._distributed():
            torch.distributed.broadcast(full, src=0)
        return self.transformer.patchify_and_maybe_split_cp(full)

    def _rank0_noise(self, like: Tensor) -> Tensor:
        """This rank's shard of one standard-normal draw over the full chunk.

        Drawn as ``(..., C, T, H, W)`` -- the reference layout -- then moved to
        ``(..., T, C, H, W)``, so the RNG stream matches the reference's.
        """
        full_shape = self._full_shape(like)
        if not self._distributed() or torch.distributed.get_rank() == 0:
            shape_cthw = (
                *full_shape[:-4],
                full_shape[-3],
                full_shape[-4],
                *full_shape[-2:],
            )
            full = (
                torch.randn(
                    shape_cthw, device=like.device, dtype=like.dtype, generator=self.rng
                )
                .movedim(-3, -4)
                .contiguous()
            )
        else:
            full = torch.empty(full_shape, device=like.device, dtype=like.dtype)
        return self._broadcast_and_shard(full)

    def _video_initial_noise(self, like: Tensor, autoregressive_index: int) -> Tensor:
        """This chunk's slice of the whole-video initial noise."""
        assert self.config.max_latent_frames is not None
        full_shape = self._full_shape(like)
        if not self._distributed() or torch.distributed.get_rank() == 0:
            if self._initial_video_noise is None:
                shape = (
                    *full_shape[:-4],
                    full_shape[-3],
                    self.config.max_latent_frames,
                    *full_shape[-2:],
                )
                # Drawn on the GPU (same generator stream as the reference),
                # parked on the CPU: a 3-minute 704x1280 video is ~0.5 GB.
                self._initial_video_noise = torch.randn(
                    shape, device=like.device, dtype=like.dtype, generator=self.rng
                ).cpu()
            start = autoregressive_index * self._len_t
            stop = start + full_shape[-4]
            if stop > self.config.max_latent_frames:
                raise ValueError(
                    f"AR step {autoregressive_index} needs latent frames "
                    f"[{start}, {stop}) but max_latent_frames is "
                    f"{self.config.max_latent_frames}"
                )
            full = (
                self._initial_video_noise[..., start:stop, :, :]
                .to(like.device)
                .movedim(-3, -4)
                .contiguous()
            )
        else:
            full = torch.empty(full_shape, device=like.device, dtype=like.dtype)
        return self._broadcast_and_shard(full)

    def generate(  # type: ignore[override]
        self,
        autoregressive_index: int,
        cache: Any,
        input: Any = None,
    ) -> tuple[Tensor, DiffusionModel.FinalState]:
        """One AR step, drawing noise in the reference layout when configured.

        Falls through to :meth:`DiffusionModel.generate` when neither knob is
        set.
        """
        if not self.config.noise_from_rank0 and self.config.max_latent_frames is None:
            return super().generate(autoregressive_index, cache, input)

        if input is not None:
            input = self.transformer.patchify_and_maybe_split_cp(input)
        cache.start(autoregressive_index)

        like = torch.empty(self.latent_shape, device=self.device, dtype=self.dtype)
        if self.config.max_latent_frames is not None:
            initial_noise = self._video_initial_noise(like, autoregressive_index)
        else:
            initial_noise = self._rank0_noise(like)
        noise_fn = self._rank0_noise if self.config.noise_from_rank0 else None

        def predict_flow(noisy_latent: Tensor, timestep: Tensor) -> Tensor:
            return self.transformer.predict_flow(
                noisy_latent=noisy_latent,
                timestep=timestep,
                cache=cache,
                input=input,
            )

        sample_kwargs: dict[str, Any] = {"noise_fn": noise_fn} if noise_fn else {}
        clean_latent = self.scheduler.sample(
            initial_noise=initial_noise,
            predict_flow=predict_flow,
            rng=self.rng,
            **sample_kwargs,
        )
        clean_latent = self.transformer.postprocess_clean_latent(
            clean_latent=clean_latent,
            cache=cache,
            input=input,
        )
        final_state = DiffusionModel.FinalState(
            clean_latent=clean_latent,
            autoregressive_index=autoregressive_index,
            cache=cache,
            input=input,
        )
        return self.transformer.unpatchify_and_maybe_gather_cp(
            clean_latent
        ), final_state

    def finalize(self, final_state: DiffusionModel.FinalState) -> None:  # type: ignore[override]
        """Commit the clean latent to the AR cache at ``t = context_noise``.

        The commit timestep is staged in the scheduler's network timestep
        dtype, the same dtype the denoising forwards see.
        """
        timestep_dtype_fn = getattr(self.scheduler, "network_timestep_dtype", None)
        timestep_dtype = (
            timestep_dtype_fn(self.dtype)
            if timestep_dtype_fn is not None
            else self.dtype
        )
        context_noise = self.config.context_noise
        timestep = torch.tensor(context_noise, device=self.device, dtype=timestep_dtype)
        if context_noise > 0.0:
            noisy_latent = self.scheduler.add_noise(
                clean_input=final_state.clean_latent,
                timestep=timestep,
                rng=self.rng,
            )
        else:
            noisy_latent = final_state.clean_latent
        self.transformer.finalize_kv_cache(
            noisy_latent=noisy_latent,
            timestep=timestep,
            cache=final_state.cache,
            input=final_state.input,
        )
        final_state.cache.finalize(final_state.autoregressive_index)


__all__ = ["EdenDiffusionModel", "EdenDiffusionModelConfig"]
