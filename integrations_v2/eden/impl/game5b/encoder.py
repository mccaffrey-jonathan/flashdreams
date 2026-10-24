# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-AR-step condition encoder for the proxy-DINO 5B model.

Two conditions are assembled here:

**First frame.** The reference inference path is ``pack_first_frame: true``: the
cover frame's own latent occupies latent frame 0 of the *first denoising chunk*
(rather than being committed as a standalone clean chunk ahead of it), is held
at ``t=0`` while the chunk's other frames denoise, and is re-pinned before every
network forward. The cover is encoded alone, matching reference independent-first-frame
conditioning. Unused later condition latents are zero; this model stamps the
first latent and does not concatenate the grey tail.

**Proxy-DINO control.** Already encoded from the rendered game frames by
:class:`~eden.impl.game14b.dino.ProxyDinoControlEncoder`; the caller supplies
the finished features, because their computation is deliberately overlapped with
the previous chunk's denoising (see
:mod:`eden.impl.game14b.control_stream`).

Both are combined into the
:class:`~eden.impl.game14b.transformer.I2VProxyDinoCtrl` payload the transformer
patchifies and channel-concatenates.

One exact saving over running the stock encoder alone: from AR step 1 on, the
mask is all zeros and this model does not channel-concat the condition latent
(``concat_image_mask_to_latent=False``), so the inner VAE's output is multiplied
out of existence by ``_stamp_image_latent`` and never reaches the network. The
VAE call is therefore skipped outright past AR 0 and a zero latent is emitted in
its place -- bit-identical output, one fewer VAE encode per chunk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from flashdreams.infra.encoder import (
    EncoderConfig,
    StreamingEncoderCache,
    StreamingVideoEncoder,
)
from flashdreams.recipes.wan.autoencoder.i2v import I2VCtrl
from flashdreams.recipes.wan.autoencoder.vae import (
    Wan22TI2V5BVAEEncoderConfig,
    WanVAECache,
    WanVAEEncoder,
    WanVAEEncoderConfig,
)

from eden.impl.game14b.transformer import I2VProxyDinoCtrl


@dataclass(kw_only=True)
class Game5BCtrlInput:
    """Per-AR-step payload handed to the encoder by the pipeline."""

    image: Tensor | None
    """Pixel chunk for the TI2V first-frame condition, ``[..., T, 3, H, W]``.

    Built by the pipeline exactly as the stock Wan I2V path builds it: the cover
    frame plus a zero tail at AR step 0, and ``None`` afterwards, where the
    condition latent is unused.
    """

    control: Tensor
    """Proxy-DINO features for this chunk, ``[..., T, 1024, Hl, Wl]``.

    Already encoded (and, in the Game2V loop, computed during the previous
    chunk's denoising) so this encoder never blocks on DINO.
    """


@dataclass(kw_only=True)
class Game5BCtrlEncoderCache(StreamingEncoderCache):
    """Per-rollout cache: the inner VAE's streaming state plus the AR-0 latent."""

    vae: WanVAECache
    """Streaming Wan 2.2 VAE cache, threaded through the AR-0 encode."""

    condition_latent: Tensor | None = None
    """The AR-0 condition latent, kept only to shape the zeros emitted later."""


@dataclass(kw_only=True)
class Game5BCtrlEncoderConfig(EncoderConfig):
    """Config for the proxy-DINO 5B condition encoder."""

    _target: type["Game5BCtrlEncoder"] = field(
        default_factory=lambda: Game5BCtrlEncoder
    )

    encoder: WanVAEEncoderConfig = field(default_factory=Wan22TI2V5BVAEEncoderConfig)
    """Streaming Wan 2.2 TI2V-5B VAE encoder; pin its checkpoint to the
    decoder's so the encoded latent matches the network's input distribution."""


class Game5BCtrlEncoder(StreamingVideoEncoder[Game5BCtrlEncoderCache]):
    """Assemble the per-chunk TI2V + proxy-DINO control payload."""

    encoder: WanVAEEncoder

    def __init__(self, config: Game5BCtrlEncoderConfig) -> None:
        super().__init__(config)
        self.config: Game5BCtrlEncoderConfig = config
        self.encoder = config.encoder.setup()

    def initialize_autoregressive_cache(self) -> Game5BCtrlEncoderCache:
        return Game5BCtrlEncoderCache(
            vae=self.encoder.initialize_autoregressive_cache(),
        )

    @torch.no_grad()
    def forward(
        self,
        input: Game5BCtrlInput,
        autoregressive_index: int = 0,
        cache: Game5BCtrlEncoderCache | None = None,
    ) -> I2VProxyDinoCtrl:
        """Build this chunk's stamp condition and pair it with the control."""
        assert cache is not None, "Game5BCtrlEncoder requires a per-rollout cache."
        control = input.control
        len_t = int(control.shape[-4])

        if autoregressive_index == 0:
            if input.image is None:
                raise ValueError(
                    "Game5BCtrlEncoder needs the cover pixels at AR step 0; the "
                    "packed first frame is encoded there and nowhere else."
                )
            latent = self.encoder(
                input.image,
                autoregressive_index=0,
                cache=cache.vae,
            )
            latent = torch.cat(
                [
                    latent,
                    latent.new_zeros(*latent.shape[:-4], len_t - 1, *latent.shape[-3:]),
                ],
                dim=-4,
            )
            cache.condition_latent = latent
            # One-hot on latent frame 0: the packed condition frame. Same shape
            # as the latent so mask and latent patchify identically and the
            # stamp is a plain elementwise blend.
            mask = torch.zeros_like(latent)
            mask[..., 0, :, :, :] = 1.0
        else:
            # Past AR 0 the mask is all zeros, so the stamp is the identity and
            # this latent is multiplied away before it reaches the network. Emit
            # zeros in the right shape instead of paying a VAE encode for a
            # tensor whose every contribution is zero.
            reference = cache.condition_latent
            if reference is None:
                raise ValueError(
                    "Game5BCtrlEncoder saw AR step "
                    f"{autoregressive_index} before AR step 0; the rollout must "
                    "start at 0 so the packed first frame is encoded."
                )
            latent = reference.new_zeros(
                *reference.shape[:-4], len_t, *reference.shape[-3:]
            )
            mask = torch.zeros_like(latent)

        return I2VProxyDinoCtrl(
            i2v=I2VCtrl(latent=latent, mask=mask),
            control=control.to(device=latent.device, dtype=latent.dtype),
        )

    @property
    def temporal_compression_ratio(self) -> int:
        return self.encoder.temporal_compression_ratio

    @property
    def spatial_compression_ratio(self) -> int:
        return self.encoder.spatial_compression_ratio

    def get_output_temporal_size(
        self, autoregressive_index: int, input_temporal_size: int
    ) -> int:
        return self.encoder.get_output_temporal_size(
            autoregressive_index, input_temporal_size
        )

    def get_input_temporal_size(
        self, autoregressive_index: int, output_temporal_size: int
    ) -> int:
        return self.encoder.get_input_temporal_size(
            autoregressive_index, output_temporal_size
        )


__all__ = [
    "Game5BCtrlEncoder",
    "Game5BCtrlEncoderCache",
    "Game5BCtrlEncoderConfig",
    "Game5BCtrlInput",
]
