# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the realtime Game2V pipeline configs from YAML.

The realtime pipelines are ``InstantiateConfig`` dataclasses (``_target`` plus
``setup``), and ``derive_config``, the app entry points and the tests all
consume the live objects, so the YAML does not replace them. It holds the knobs
a deployment changes — checkpoint, chunk size, attention window, frame size,
sampling schedule — and this module folds those values into the dataclasses the
rest of the code already uses. That is the same split the offline configs make
between ``eden/configs/*.yaml`` and the adapter that reads them, except the
result here is a config object rather than a plain dict.

Paths under ``components`` are relative to the config's own ``model_root``, or
to :data:`eden.model_root.EDEN_MODEL_ROOT` when that is ``null``, so no
machine-local prefix is baked into the code. Absolute paths and URLs are used
as written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from eden.model_root import eden_model_root

CONFIGS_DIR = Path(__file__).parent / "configs"

# ``timestep_dtype`` is a torch dtype, which YAML cannot spell, so it is named.
_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int64": torch.int64,
}


def load_pipeline_yaml(name: str) -> dict[str, Any]:
    """Read ``configs/<name>.yaml`` next to this module."""
    path = CONFIGS_DIR / f"{name}.yaml"
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a mapping, got {type(loaded).__name__}")
    return loaded


def model_root(spec: dict[str, Any]) -> str:
    """The weight tree this config reads from.

    The config's own ``model_root`` wins; ``null`` (or absent) falls back to
    ``EDEN_MODEL_ROOT`` in the environment.
    """
    root = spec.get("model_root")
    if isinstance(root, str) and root:
        return root.rstrip("/")
    return eden_model_root().rstrip("/")


def resolve_path(spec: dict[str, Any], raw: str) -> str:
    """Resolve one config path against ``spec``'s model root.

    Absolute paths and URLs pass through unchanged.
    """
    if raw.startswith(("http://", "https://")) or Path(raw).is_absolute():
        return raw
    return f"{model_root(spec)}/{raw}"


def component_path(spec: dict[str, Any], key: str) -> str:
    """Resolve ``components.<key>`` against the config's model root."""
    components = spec.get("components") or {}
    raw = components.get(key)
    if not isinstance(raw, str) or not raw:
        raise KeyError(f"components.{key} is missing from the pipeline config")
    return resolve_path(spec, raw)


def _dtype(name: str, value: str) -> torch.dtype:
    try:
        return _DTYPES[value]
    except KeyError:
        raise ValueError(
            f"unknown {name} {value!r}; expected one of {sorted(_DTYPES)}"
        ) from None


def scheduler_kwargs(spec: dict[str, Any]) -> dict[str, Any]:
    """The ``scheduler`` block, with its dtype names turned into dtypes."""
    block = dict(spec.get("scheduler") or {})
    for name in ("timestep_dtype", "network_timestep_dtype"):
        value = block.get(name)
        if isinstance(value, str):
            block[name] = _dtype(name, value)
    return block
