# SPDX-FileCopyrightText: Copyright (c) 2026 Vast AI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for FlashDreams v0.2.0: the EdenLive fork patched
# flashdreams/infra/diffusion/scheduler/{base,fm}.py in place; the same
# behaviour lives here as a subclass so core numerics stay unchanged.

"""Flow-match scheduler with the EdenLive reference sampling arithmetic.

Three differences from :class:`FlowMatchScheduler`, all needed to reproduce the
EdenLive 4-step sampler (VAST's offline package and LightX2V's
``wan22_edenlive_ar`` runner):

* **fp32 network timesteps.** The warped grid is ``[1000, 937.5, 833.33, 625]``;
  casting those to the bf16 latent dtype rounds them to 936 / 832 / 624 while the
  ``x0`` conversion still uses the exact sigma. ``network_timestep_dtype`` keeps
  them fp32 (and :class:`~eden.impl.diffusion.EdenDiffusionModel` stages the
  cache-commit timestep in the same dtype, so one CUDA graph serves both).
* **bf16 sigma.** ``x0 = x - sigma * v`` and the re-noise lerp run in the latent
  dtype, as the reference does, instead of auto-promoting to fp32.
* **Injected noise.** ``sample(noise_fn=...)`` draws the re-noise through the
  diffusion model, which lays it out in the reference's ``(C, T, H, W)`` order.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor
from tqdm import tqdm

from flashdreams.infra.diffusion.scheduler.base import FlowPredictor
from flashdreams.infra.diffusion.scheduler.fm import (
    FlowMatchScheduler,
    FlowMatchSchedulerConfig,
)

NoiseFn = Callable[[Tensor], Tensor]
"""``like -> noise``: a standard-normal tensor shaped like ``like``."""


@dataclass(kw_only=True)
class EdenFlowMatchSchedulerConfig(FlowMatchSchedulerConfig):
    """:class:`FlowMatchSchedulerConfig` plus the network timestep dtype."""

    _target: type["EdenFlowMatchScheduler"] = field(
        default_factory=lambda: EdenFlowMatchScheduler
    )

    network_timestep_dtype: torch.dtype | None = None
    """Dtype of the timestep passed to ``predict_flow``. ``None`` casts it to
    the latent dtype. ``torch.float32`` keeps fractional warped timesteps
    exact."""


class EdenFlowMatchScheduler(FlowMatchScheduler):
    """Self-forcing flow-match loop with EdenLive's reference arithmetic."""

    config: EdenFlowMatchSchedulerConfig

    def __init__(self, config: EdenFlowMatchSchedulerConfig) -> None:
        if config.reference_sampling:
            raise ValueError(
                "EdenFlowMatchScheduler implements its own reference arithmetic; "
                "leave reference_sampling (the LingBot mode) off"
            )
        super().__init__(config)
        self.config = config

    def network_timestep_dtype(self, input_dtype: torch.dtype) -> torch.dtype:
        """Dtype of the timestep handed to ``predict_flow`` for ``input_dtype``."""
        configured = self.config.network_timestep_dtype
        return input_dtype if configured is None else configured

    def sample(  # type: ignore[override]
        self,
        initial_noise: Tensor,
        predict_flow: FlowPredictor,
        rng: torch.Generator | None = None,
        noise_fn: NoiseFn | None = None,
    ) -> Tensor:
        """Run the 4-step loop: ``x0 = x - sigma v``, re-noise, repeat.

        Iteration 0 trusts ``initial_noise`` as the ``sigma=1`` sample; later
        iterations re-noise the previous ``x0`` to the next sigma. ``noise_fn``,
        when given, draws that re-noise instead of ``rng``.
        """
        input_dtype = initial_noise.dtype
        timestep_dtype = self.network_timestep_dtype(input_dtype)
        num_train = self.config.num_train_timesteps
        sigmas = self.denoising_sigmas
        timesteps = self.denoising_step_list

        noisy = initial_noise
        clean: Tensor | None = None
        for i in tqdm(
            range(timesteps.shape[0]),
            disable=not self.config.enable_tqdm,
            desc="EdenFlowMatchScheduler",
        ):
            sigma = sigmas[i].to(dtype=input_dtype)
            # The reference normalises the explicit schedule to sigma on the
            # GPU and rebuilds the network timestep from it; keep that fp32
            # round-trip so the embedded value matches bit for bit.
            timestep = (timesteps[i] / num_train * num_train).to(dtype=timestep_dtype)
            if i > 0:
                assert clean is not None
                noise = (
                    noise_fn(noisy)
                    if noise_fn is not None
                    else torch.empty_like(noisy).normal_(generator=rng)
                )
                noisy = ((1.0 - sigma) * clean + sigma * noise).to(input_dtype)
            flow = predict_flow(noisy, timestep)
            clean = noisy - sigma * flow
        assert clean is not None, "denoising_step_list is empty"
        return clean.to(input_dtype)


__all__ = [
    "EdenFlowMatchScheduler",
    "EdenFlowMatchSchedulerConfig",
    "NoiseFn",
]
