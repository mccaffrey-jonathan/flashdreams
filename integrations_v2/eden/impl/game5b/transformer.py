# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan 2.2 TI2V-5B transformer carrying the proxy-DINO control channels.

Adds two things to :class:`Wan21Transformer`:

* A composite per-AR-step payload (:class:`I2VProxyDinoCtrl`, shared with the
  14B variant) that travels the TI2V first-frame latent/mask alongside the
  1024-channel proxy-DINO control, so both patchify together and the control is
  appended after the 48 latent channels -- the channel order the checkpoint was
  trained on.
* A DCP checkpoint load, since the training run saves an FSDP2 distributed
  checkpoint with Diffusers module names and four interleaved module trees. The
  base class's ``checkpoint_path`` cannot express that (no key remap, no
  subtree selection), so loading happens here and ``checkpoint_path`` is left
  unset on the config.

Unlike the 14B AR variant there is no control RMSNorm to apply: this run's
patch embedding is a plain ``nn.Conv3d`` (see :mod:`eden.impl.game5b.network`),
so patchify is the base implementation applied to each part of the payload.

First-frame conditioning is the reference's ``pack_first_frame: true``: the
condition latent sits inside the first denoising chunk at latent frame 0, held
at ``t=0`` while frames 1: denoise at the scheduler step, and re-pinned before
every network forward. That is exactly the inherited Wan 2.2 TI2V machinery --
``stamp_image_latent`` re-injects it ahead of each forward,
``ti2v_first_frame_per_token_timestep`` gives its tokens ``t=0``, and
``postprocess_clean_latent`` re-stamps the final ``x0`` -- so no override is
needed here beyond forwarding the I2V half of the payload to it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from loguru import logger

from flashdreams.recipes.wan.transformer.wan21 import (
    Wan21Transformer,
    Wan21TransformerConfig,
)

# The payload is structurally identical to the 14B's, so it is shared rather
# than redeclared; ``eden.impl.game14b`` hosts the proxy-DINO pieces both
# models use (this one, the DINOv3 encoder and the overlapped control stream).
from eden.impl.game14b.transformer import I2VProxyDinoCtrl

from .checkpoint import load_game5b_dit_state_dict
from .network import Game5BDiTNetwork, Game5BDiTNetworkConfig


@dataclass(kw_only=True)
class Game5BTransformerConfig(Wan21TransformerConfig):
    """Config for the proxy-DINO 5B transformer."""

    _target: type["Game5BTransformer"] = field(
        default_factory=lambda: Game5BTransformer
    )

    network: Game5BDiTNetworkConfig = field(default_factory=Game5BDiTNetworkConfig)

    dcp_checkpoint_path: str | None = None
    """Published safetensors file, holding the distilled EMA with Diffusers keys.

    Loaded after construction with the Diffusers key remap. Distinct from the
    base ``checkpoint_path``, which only handles files whose keys already match
    the network.
    """


class Game5BTransformer(Wan21Transformer):
    """TI2V-5B transformer with proxy-DINO channel-concat control."""

    config: Game5BTransformerConfig
    network: Game5BDiTNetwork

    def __init__(self, config: Game5BTransformerConfig) -> None:
        if config.dcp_checkpoint_path is not None and config.checkpoint_path is not None:
            raise ValueError(
                "set either dcp_checkpoint_path or checkpoint_path, not both"
            )
        super().__init__(config)
        self.config = config

    @staticmethod
    def _setup_network(config: Game5BTransformerConfig) -> Game5BDiTNetwork:
        """Build the network and load the weights before compile.

        Loading here rather than in ``__init__`` keeps it ahead of
        ``torch.compile`` and the CUDA-graph wrappers the base constructor
        installs -- both capture ``self.network`` by reference, so swapping
        parameters afterwards would leave them pointing at stale storage. It
        also lands before the base's
        ``update_parameters_after_loading_checkpoint``, whose weight fusions
        must observe the real checkpoint values.
        """
        network = Wan21Transformer._setup_network(config)
        assert isinstance(network, Game5BDiTNetwork), (
            f"expected a Game5BDiTNetwork, got {type(network).__name__}"
        )
        if config.dcp_checkpoint_path is None:
            return network
        # The heaviest single step of startup: the whole 5B DiT off a network
        # filesystem.
        logger.info(
            "Loading Game2V 5B DiT weights from {}", config.dcp_checkpoint_path
        )
        load_started = time.perf_counter()
        state_dict = load_game5b_dit_state_dict(config.dcp_checkpoint_path)
        # The checkpoint stores fp32; the network runs at ``config.dtype``.
        state_dict = {
            key: value.to(dtype=config.dtype) for key, value in state_dict.items()
        }
        network.load_state_dict(state_dict, strict=True, assign=True)
        logger.info(
            "Game2V 5B DiT weights loaded in {:.1f}s ({} tensors)",
            time.perf_counter() - load_started,
            len(state_dict),
        )
        return network

    def patchify_and_maybe_split_cp(self, x):  # type: ignore[override]
        """Patchify both halves of the composite payload.

        No control normalization here, unlike the 14B: this run's patch
        embedding is a plain convolution over the concatenated channels.
        """
        if isinstance(x, I2VProxyDinoCtrl):
            if x._is_patchified:
                return x
            return I2VProxyDinoCtrl(
                i2v=super().patchify_and_maybe_split_cp(x.i2v),
                control=super().patchify_and_maybe_split_cp(x.control),
                _is_patchified=True,
            )
        return super().patchify_and_maybe_split_cp(x)

    def _build_network_input(  # type: ignore[override]
        self,
        noisy_latent: torch.Tensor,
        input: I2VProxyDinoCtrl | None,
    ) -> torch.Tensor:
        """Concatenate ``[noisy+stamp 48 | proxy-DINO 1024]``.

        The TI2V half is delegated to the base implementation, which re-injects
        the clean first-frame latent at the masked positions (the
        ``pack_first_frame`` pin); the control is appended last.
        """
        assert isinstance(input, I2VProxyDinoCtrl), (
            "Game5BTransformer requires an I2VProxyDinoCtrl payload, got "
            f"{type(input).__name__}"
        )
        network_input = super()._build_network_input(noisy_latent, input.i2v)
        return torch.cat([network_input, input.control], dim=-1)

    def postprocess_clean_latent(  # type: ignore[override]
        self,
        clean_latent: torch.Tensor,
        cache,
        input: I2VProxyDinoCtrl | None = None,
    ) -> torch.Tensor:
        """Forward the I2V payload to the base stamp re-injection."""
        return super().postprocess_clean_latent(
            clean_latent=clean_latent,
            cache=cache,
            input=input.i2v if input is not None else None,
        )

    def _maybe_build_per_token_timestep(  # type: ignore[override]
        self,
        timestep: torch.Tensor,
        input: I2VProxyDinoCtrl | None,
        autoregressive_index: int,
    ) -> torch.Tensor:
        """Forward the I2V payload to the base per-token timestep hook.

        This is what holds the packed first frame at ``t=0`` through AR step 0
        while the rest of the chunk denoises at the scheduler timestep.
        """
        return super()._maybe_build_per_token_timestep(
            timestep=timestep,
            input=input.i2v if input is not None else None,
            autoregressive_index=autoregressive_index,
        )


__all__ = [
    "Game5BTransformer",
    "Game5BTransformerConfig",
    "I2VProxyDinoCtrl",
]
