# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint loading for the proxy-DINO 5B DiT.

The published weights are the distilled DiT, saved as a single safetensors
file with its Diffusers key names.

The key surface needs nothing beyond the shared Diffusers remap: this model
has no CLIP cross-attention branch (Wan 2.2 TI2V 5B never had one) and its
channel-concat patch embedding is a plain ``nn.Conv3d``, so there is no
``patch_embedding.condition_norm`` to place. Verified against the file: all
825 remapped keys match
:class:`~flashdreams.recipes.wan.transformer.impl.network.WanDiTNetwork`
exactly, with no extras and no shape mismatches.
"""

from __future__ import annotations

import torch
from safetensors.torch import load_file

from flashdreams.recipes.wan.transformer.checkpoint import (
    wan_dit_state_dict_from_diffusers,
)


def game5b_dit_state_dict_from_diffusers(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remap a proxy-DINO 5B Diffusers state dict to FlashDreams keys."""
    return wan_dit_state_dict_from_diffusers(state_dict)


def load_game5b_dit_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    """Read the DiT weights out of the published safetensors file.

    Args:
        checkpoint_path: ``dit-5b/diffusion_pytorch_model.safetensors``.

    Returns:
        A state dict keyed for :class:`WanDiTNetwork`.
    """
    return game5b_dit_state_dict_from_diffusers(load_file(checkpoint_path))


__all__ = [
    "game5b_dit_state_dict_from_diffusers",
    "load_game5b_dit_state_dict",
]
