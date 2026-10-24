# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan 2.1 transformer carrying the proxy-DINO control channels.

Adds two things to :class:`Wan21Transformer`:

* A composite per-AR-step payload (:class:`I2VProxyDinoCtrl`) that travels
  the I2V latent/mask alongside the 1024-channel proxy-DINO control, so all
  three patchify together and the control is appended after the 36 I2V
  channels — the exact order the checkpoint was trained on.
* A DCP checkpoint load, since the training run saves an FSDP2 distributed
  checkpoint with Diffusers module names. The base class's
  ``checkpoint_path`` cannot express that (no key remap), so loading happens
  here and ``checkpoint_path`` is left unset on the config.

The control RMSNorm runs during patchify (once per AR chunk) rather than
inside ``predict_flow`` (once per denoising step); see
:mod:`eden.impl.game14b.network`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from loguru import logger
from torch import Tensor

from flashdreams.recipes.wan.autoencoder.i2v import I2VCtrl
from flashdreams.recipes.wan.transformer.wan21 import (
    Wan21Transformer,
    Wan21TransformerConfig,
)

from .checkpoint import load_game14b_dit_state_dict
from .network import Game14BDiTNetwork, Game14BDiTNetworkConfig


@dataclass(kw_only=True)
class I2VProxyDinoCtrl:
    """Per-AR-step I2V control plus the proxy-DINO channel-concat control."""

    i2v: I2VCtrl
    """First-frame mask + condition latent (the 20 trailing I2V channels)."""

    control: Tensor
    """Proxy-DINO features, ``[..., T, 1024, Hl, Wl]`` before patchify."""

    _is_patchified: bool = False


@dataclass(kw_only=True)
class Game14BTransformerConfig(Wan21TransformerConfig):
    """Config for the proxy-DINO AR 14B transformer."""

    _target: type["Game14BTransformer"] = field(
        default_factory=lambda: Game14BTransformer
    )

    network: Game14BDiTNetworkConfig = field(
        default_factory=Game14BDiTNetworkConfig
    )

    dcp_checkpoint_path: str | None = None
    """Published safetensors file, holding the DiT with Diffusers keys.

    Loaded after construction with the Diffusers key remap. Distinct from
    the base ``checkpoint_path``, which only handles files whose keys
    already match the network.
    """


class Game14BTransformer(Wan21Transformer):
    """14B I2V transformer with proxy-DINO channel-concat control."""

    config: Game14BTransformerConfig
    network: Game14BDiTNetwork

    def __init__(self, config: Game14BTransformerConfig) -> None:
        if config.dcp_checkpoint_path is not None and config.checkpoint_path is not None:
            raise ValueError(
                "set either dcp_checkpoint_path or checkpoint_path, not both"
            )
        super().__init__(config)
        self.config = config

    @staticmethod
    def _setup_network(config: Game14BTransformerConfig) -> Game14BDiTNetwork:
        """Build the network and load the weights before compile.

        Loading here rather than in ``__init__`` keeps it ahead of
        ``torch.compile`` and the CUDA-graph wrappers the base constructor
        installs — both capture ``self.network`` by reference, so swapping
        parameters afterwards would leave them pointing at stale storage.
        It also lands before the base's
        ``update_parameters_after_loading_checkpoint``, whose weight fusions
        must observe the real checkpoint values.
        """
        network = Wan21Transformer._setup_network(config)
        assert isinstance(network, Game14BDiTNetwork), (
            f"expected a Game14BDiTNetwork, got {type(network).__name__}"
        )
        if config.dcp_checkpoint_path is not None:
            # The heaviest single step of startup: the whole 14B DiT off a
            # network filesystem.
            logger.info(
                "Loading Game2V 14B DiT weights from {}", config.dcp_checkpoint_path
            )
            load_started = time.perf_counter()
            state_dict = load_game14b_dit_state_dict(config.dcp_checkpoint_path)
            state_dict = {
                key: value.to(dtype=config.dtype)
                for key, value in state_dict.items()
            }
            network.load_state_dict(state_dict, strict=True, assign=True)
            logger.info(
                "Game2V 14B DiT weights loaded in {:.1f}s ({} tensors)",
                time.perf_counter() - load_started,
                len(state_dict),
            )
        return network

    def _unwrapped_network(self) -> Game14BDiTNetwork:
        """Return the eager module behind any ``torch.compile`` wrapper."""
        network = self.network
        inner = getattr(network, "_orig_mod", network)
        assert isinstance(inner, Game14BDiTNetwork), (
            f"expected a Game14BDiTNetwork, got {type(inner).__name__}"
        )
        return inner

    def patchify_and_maybe_split_cp(self, x):  # type: ignore[override]
        """Patchify the composite payload, normalizing the control first.

        The RMSNorm is applied here, on the pre-patchify ``[..., T, C, H, W]``
        control, so it sees the channel axis the reference normalizes and
        runs once per AR chunk rather than once per denoising step.
        """
        if isinstance(x, I2VProxyDinoCtrl):
            if x._is_patchified:
                return x
            control = self._unwrapped_network().normalize_control_channels(x.control)
            return I2VProxyDinoCtrl(
                i2v=super().patchify_and_maybe_split_cp(x.i2v),
                control=super().patchify_and_maybe_split_cp(control),
                _is_patchified=True,
            )
        return super().patchify_and_maybe_split_cp(x)

    def _build_network_input(  # type: ignore[override]
        self,
        noisy_latent: Tensor,
        input: I2VProxyDinoCtrl | None,
    ) -> Tensor:
        """Concatenate ``[noisy | mask | image latent | proxy-DINO]``.

        The I2V part is delegated to the base implementation so the mask's
        16-channel encoder tag is sliced to 4 exactly as upstream does; the
        already-normalized control is appended last.
        """
        assert isinstance(input, I2VProxyDinoCtrl), (
            "Game14BTransformer requires an I2VProxyDinoCtrl payload, got "
            f"{type(input).__name__}"
        )
        network_input = super()._build_network_input(noisy_latent, input.i2v)
        return torch.cat([network_input, input.control], dim=-1)

    def postprocess_clean_latent(  # type: ignore[override]
        self,
        clean_latent: Tensor,
        cache,
        input: I2VProxyDinoCtrl | None = None,
    ) -> Tensor:
        """Forward the I2V payload to the base mask-inject re-stamp."""
        return super().postprocess_clean_latent(
            clean_latent=clean_latent,
            cache=cache,
            input=input.i2v if input is not None else None,
        )

    def _maybe_build_per_token_timestep(  # type: ignore[override]
        self,
        timestep: Tensor,
        input: I2VProxyDinoCtrl | None,
        autoregressive_index: int,
    ) -> Tensor:
        """Forward the I2V payload to the base per-token timestep hook."""
        return super()._maybe_build_per_token_timestep(
            timestep=timestep,
            input=input.i2v if input is not None else None,
            autoregressive_index=autoregressive_index,
        )


__all__ = [
    "Game14BTransformer",
    "Game14BTransformerConfig",
    "I2VProxyDinoCtrl",
]
