# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Game2V wiring for the proxy-DINO conditioned AR 14B model.

Three things differ from the camera-control presets:

* **First frame.** Each game's own ``cover`` image from ``games/registry.json``
  rather than one shared example frame.
* **Prompt.** The registry's static ``prompt`` for that game, encoded once at
  startup, after which the text encoder is released. ``--dynamic-prompt`` keeps
  the encoder resident on rank zero and re-encodes each chunk's prompt — the
  static text plus the event lines still inside their lifetime — through
  :meth:`~eden.impl.game14b.pipeline.Game14BInferencePipeline.encode_text`.
* **Step input.** The encoded render, not a camera payload. The encode is
  overlapped with the previous chunk's denoising by
  :class:`~eden.impl.game14b.control_stream.ProxyDinoControlStream`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from cam2v import Cam2VConditioning

from eden.impl.game14b.control_stream import ProxyDinoControlStream
from eden.impl.game14b.dino import ProxyDinoEncoderConfig

def _games_root() -> Path:
    """Locate the vendored games directory the renderer also reads."""
    return Path(
        os.environ.get(
            "EDEN_GAMES_ROOT", Path(__file__).resolve().parents[4] / "games"
        )
    ).resolve()


def _registry_entry(slug: str) -> Mapping[str, Any]:
    root = _games_root()
    entries = json.loads((root / "registry.json").read_text())["games"]
    for entry in entries:
        if entry.get("slug") == slug:
            return entry
    raise ValueError(f"Unregistered game: {slug}")


def game_cover_path(slug: str) -> Path:
    """Return the cover image the game's registry entry declares."""
    entry = _registry_entry(slug)
    cover = entry.get("cover")
    if not cover:
        raise ValueError(f"Game {slug!r} has no 'cover' in its registry entry.")
    path = _games_root() / slug / str(cover)
    if not path.is_file():
        raise FileNotFoundError(f"Game {slug!r} cover image is missing: {path}")
    return path


def resolve_game14b_conditioning(values: Mapping[str, Any]) -> Cam2VConditioning:
    """Resolve the first frame and prompt from the selected game's registry.

    ``--image-path`` and ``--prompt`` still override, so a single game can be
    driven from an arbitrary frame or prompt without touching the registry.
    """
    slug = str(values.get("game") or "")
    if not slug:
        raise ValueError("Game2V 14B conditioning requires a game slug.")
    entry = _registry_entry(slug)

    image_path = values.get("image_path")
    first_frame_path = (
        Path(image_path) if image_path not in (None, "") else game_cover_path(slug)
    )
    if not first_frame_path.is_file():
        raise FileNotFoundError(f"Game2V first frame is missing: {first_frame_path}")

    prompt = values.get("prompt") or ""
    if not prompt:
        prompt = str(entry.get("prompt", "")).strip()

    pixel_width = int(values.get("pixel_width", 832))
    pixel_height = int(values.get("pixel_height", 480))
    intrinsics = torch.tensor(
        [
            0.5 * pixel_width,
            0.5 * pixel_height,
            0.5 * pixel_width,
            0.5 * pixel_height,
        ],
        dtype=torch.float32,
    )
    return Cam2VConditioning(
        prompt=prompt,
        first_frame_path=first_frame_path,
        base_intrinsics=intrinsics,
        # Camera control is inactive for this model; the pose branch is never
        # read, so the scale is a placeholder the contract still requires.
        world_scale=1.0,
    )


def generate_game14b_step(
    pipeline: Any,
    autoregressive_index: int,
    cache: Any,
    control: Any,
) -> torch.Tensor:
    """Generate one chunk from the already-encoded proxy-DINO control."""
    return pipeline.generate(
        autoregressive_index=autoregressive_index,
        cache=cache,
        input=control,
    )


def make_game14b_control_stream(config: Any, session_desc: Any) -> ProxyDinoControlStream:
    """Build the overlapped proxy-DINO encoder for one session.

    Latent dimensions come from the model's own frame size divided by the Wan
    VAE's 8x spatial ratio, so the features land on the DiT's grid without a
    second resample at consume time.
    """
    from eden.game14b_config import GAME2V_LEN_T, _SPEC
    from eden.pipeline_yaml import component_path

    pixel_width = config.model_video_width or session_desc.video_width
    pixel_height = config.model_video_height or session_desc.video_height
    spatial_ratio = 8
    return ProxyDinoControlStream(
        ProxyDinoEncoderConfig(
            backbone_repo=component_path(_SPEC, "vision_encoder"),
            projection_path=component_path(_SPEC, "vision_projection"),
            resize_mode="bicubic_center_crop",
            device=str(config.device),
            compile_encoder=getattr(config, "compile_control_encoder", True),
        ),
        latent_height=pixel_height // spatial_ratio,
        latent_width=pixel_width // spatial_ratio,
        len_t=GAME2V_LEN_T,
        max_latent_frames=_SPEC["max_latent_frames"],
    )


__all__ = [
    "game_cover_path",
    "generate_game14b_step",
    "make_game14b_control_stream",
    "resolve_game14b_conditioning",
]
