# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration-owned defaults and resolved inputs for camera-to-video apps."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
from cam2v import Cam2VConditioning

from flashdreams.infra.runner_io import ResizeInterpolation
from flashdreams.runtime_v2.session_desc import BackpressureMode, PresentationMode
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

from .controls import CameraPoseIntegrator

Cam2VInputResolver = Callable[[Mapping[str, Any]], "Cam2VConditioning"]
"""Resolve application arguments into one session's camera conditioning."""

Cam2VGenerateStep = Callable[[Any, int, Any, Any], torch.Tensor]
"""Generate one model step from a model-neutral camera payload."""


def generate_camera_step(
    pipeline: Any,
    autoregressive_index: int,
    cache: Any,
    camera_input: Any,
) -> torch.Tensor:
    """Generate one camera-controlled step through the standard pipeline API."""
    return pipeline.generate(
        autoregressive_index=autoregressive_index,
        cache=cache,
        input=camera_input,
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class Cam2VApplicationDefaults:
    """Defaults one model integration contributes to the shared Cam2V app."""

    pipeline_config: Any
    """Model pipeline configuration owned by the integration."""

    input_resolver: Cam2VInputResolver
    """Integration hook that resolves paths and camera calibration."""

    total_blocks: int
    """Default number of autoregressive blocks in one rollout."""

    pixel_width: int
    """Default generated frame width."""

    pixel_height: int
    """Default generated frame height."""

    first_frame_dtype: torch.dtype
    """Tensor dtype used for the model's normalized first frame."""

    first_frame_interpolation: ResizeInterpolation
    """Resize interpolation required by the model's image preprocessor."""

    generate_step: Cam2VGenerateStep = generate_camera_step
    """Integration hook that adapts camera payloads and generates one step."""

    pose_integrator_factory: Callable[[], CameraPoseIntegrator] = CameraPoseIntegrator
    """Construct the integration-calibrated camera pose integrator."""

    device: str = "cuda"
    """Device on which the application constructs the shared pipeline."""

    fps: int = 16
    """Initial video frame rate and model-generation-loop pacing limit."""

    output_layout: VideoTensorLayout = VideoTensorLayout.tchw
    """Tensor layout emitted by the model pipeline."""

    backpressure_mode: BackpressureMode = BackpressureMode.BLOCK
    """Preserve every generated model frame in presentation order."""

    presentation_mode: PresentationMode = PresentationMode.CONTINUOUS
    """Render the UI every tick so controls and status remain responsive."""

    ui_fps: int = 60
    """Rate at which the UI thread reads inputs and runs the UI loop."""

    warmup_blocks: int = 5
    """Leading blocks excluded from steady-state FPS."""

    log_model_timing: bool = False
    """Write one synchronized wall-time record for each AR model step."""

    install_hint: str = ""
    """Optional dependency hint included in first-frame loading failures."""

    input_defaults: Mapping[str, Any] = field(default_factory=dict)
    """Integration-owned default prompt, asset paths, and example selection."""

    fixed_camera: bool = True
    """Keep the initial camera pose while the game input is not connected."""

    game_slug: str = "freeze"
    """Vendored WebGameVault game used by the Game2V timeline."""

    control_stream_factory: Callable[[Any, Any], Any] | None = None
    """Build the overlapped condition encoder for frame-conditioned models.

    ``None`` keeps the camera-control behavior: steps are driven by poses and
    nothing encodes the rendered frames. Models that condition on the render
    supply a factory here; the app threads it into the session config.
    """

    def __post_init__(self) -> None:
        if self.total_blocks <= 0:
            raise ValueError("Cam2VApplicationDefaults.total_blocks must be > 0.")
        if self.pixel_width <= 0 or self.pixel_height <= 0:
            raise ValueError("Cam2VApplicationDefaults dimensions must be > 0.")
        if self.fps <= 0 or self.ui_fps <= 0:
            raise ValueError("Cam2VApplicationDefaults frame rates must be > 0.")
        if self.warmup_blocks < 0:
            raise ValueError("Cam2VApplicationDefaults.warmup_blocks must be >= 0.")
        if not isinstance(self.log_model_timing, bool):
            raise TypeError("Cam2VApplicationDefaults.log_model_timing must be bool.")
        object.__setattr__(
            self,
            "input_defaults",
            MappingProxyType(dict(self.input_defaults)),
        )


__all__ = [
    "Cam2VApplicationDefaults",
    "Cam2VConditioning",
    "Cam2VGenerateStep",
    "Cam2VInputResolver",
    "generate_camera_step",
]
