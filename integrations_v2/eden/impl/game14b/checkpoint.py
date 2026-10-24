# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint loading for the AR teacher-forced 14B proxy-DINO DiT.

The published file keeps the training checkpoint's Diffusers module names,
with the ``model.`` prefix that FSDP2 added already stripped. One transform
gets it into FlashDreams' ``WanDiTNetwork`` layout: the shared Diffusers
remap, extended with the three groups the base table does not cover
(verified against the file's own tensor shapes):

- ``attn2.add_k_proj`` / ``add_v_proj`` / ``norm_added_k`` — the Wan 2.1
  I2V CLIP cross-attention branch, which FlashDreams names
  ``cross_attn.k_img`` / ``v_img`` / ``norm_k_img``.
- ``condition_embedder.image_embedder.*`` — the CLIP ``MLPProj``, whose
  ``LayerNorm/Linear/GELU/Linear/LayerNorm`` stack lands on
  ``img_emb.proj.{0,1,3,4}``.
- ``patch_embedding.condition_norm.weight`` — the affine RMSNorm over
  the appended proxy-DINO control channels. It passes through unchanged,
  but the module has to exist before loading, which is why the network
  subclass builds it eagerly.
"""

from __future__ import annotations

import torch
from safetensors.torch import load_file

from flashdreams.core.checkpoint.remap import remap_checkpoint_keys
from flashdreams.recipes.wan.transformer.checkpoint import (
    wan_dit_state_dict_from_diffusers,
)

# Applied after ``wan_dit_state_dict_from_diffusers``; these keys have no
# entry in the shared table.
_PROXY_DINO_KEY_REMAP: dict[str, str] = {
    # Wan 2.1 I2V CLIP cross-attention branch.
    r"^blocks\.(\d+)\.attn2\.add_k_proj\.(.*)$": r"blocks.\1.cross_attn.k_img.\2",
    r"^blocks\.(\d+)\.attn2\.add_v_proj\.(.*)$": r"blocks.\1.cross_attn.v_img.\2",
    r"^blocks\.(\d+)\.attn2\.norm_added_k\.(.*)$": r"blocks.\1.cross_attn.norm_k_img.\2",
    # CLIP image embedder -> MLPProj sequential. Index 2 is the GELU.
    r"^condition_embedder\.image_embedder\.norm1\.(.*)$": r"img_emb.proj.0.\1",
    r"^condition_embedder\.image_embedder\.ff\.net\.0\.proj\.(.*)$": r"img_emb.proj.1.\1",
    r"^condition_embedder\.image_embedder\.ff\.net\.2\.(.*)$": r"img_emb.proj.3.\1",
    r"^condition_embedder\.image_embedder\.norm2\.(.*)$": r"img_emb.proj.4.\1",
}


def game14b_dit_state_dict_from_diffusers(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remap a proxy-DINO AR 14B Diffusers state dict to FlashDreams keys."""
    return remap_checkpoint_keys(
        wan_dit_state_dict_from_diffusers(state_dict),
        _PROXY_DINO_KEY_REMAP,
    )


def load_game14b_dit_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    """Read the DiT weights out of the published safetensors file.

    Args:
        checkpoint_path: ``dit-14b/diffusion_pytorch_model.safetensors``.
    """
    return game14b_dit_state_dict_from_diffusers(load_file(checkpoint_path))


__all__ = [
    "game14b_dit_state_dict_from_diffusers",
    "load_game14b_dit_state_dict",
]
