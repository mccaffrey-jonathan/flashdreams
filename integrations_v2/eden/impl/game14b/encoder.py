# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-AR-step condition encoder for the proxy-DINO AR 14B model.

Two conditions are assembled here:

**First frame.** The reference uses ``masked_video``: encode the cover at
pixel frame 0 followed by zero pixels through one causal VAE stream. The tail
retains the cover's temporal influence. Its global time continues advancing
when the attention arena fills; RoPE slots do not change condition time.

The mask marks latent frame 0 of the rollout only (AR step 0, frame 0). The
same frame is stamped into the noisy latent and held at ``t=0`` by the
transformer, which is the ``pack_first_frame`` pin.

**Proxy-DINO control.** Already encoded from the rendered game frames by
:class:`~eden.impl.game14b.dino.ProxyDinoControlEncoder`; the caller supplies
the finished features, because their computation is deliberately overlapped
with the previous chunk's denoising.

Both are combined into the :class:`~eden.impl.game14b.transformer.I2VProxyDinoCtrl`
payload the transformer patchifies and channel-concatenates.
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
    WanVAECache,
    WanVAEEncoder,
    WanVAEEncoderConfig,
)

from .transformer import I2VProxyDinoCtrl

# Channels the mask occupies in the network input. The encoder emits a
# 16-channel uniform tag so mask and latent patchify identically; the
# transformer slices the leading 16 patch entries back to the 4 real channels.
_MASK_TAG_CHANNELS = 16


@dataclass(kw_only=True)
class Game14BCtrlInput:
    """Per-AR-step payload handed to the encoder by the pipeline."""

    image: Tensor | None
    """Cover frame ``[..., 1, 3, H, W]`` in ``[-1, 1]`` at AR step 0, else ``None``.

    The cover is encoded at AR step 0 and nowhere else; later steps only read
    the cached grey tail.
    """

    control: Tensor
    """Proxy-DINO features for this chunk, ``[..., T, 1024, Hl, Wl]``.

    Already encoded (and, in the Game2V loop, computed during the previous
    chunk's denoising) so this encoder never blocks on DINO.
    """


@dataclass(kw_only=True)
class Game14BCtrlEncoderCache(StreamingEncoderCache):
    """Per-rollout cache: the cover latent and the grey tail it is spliced onto."""

    vae: WanVAECache
    """Streaming cache for the cover followed by zero pixels."""

    cover_latent: Tensor | None = None
    """``[..., 1, 16, Hl, Wl]`` latent of the cover encoded alone (AR step 0)."""

    grey_tail: Tensor | None = None
    """Most recent masked-video condition chunk; used to recover pixel geometry."""


@dataclass(kw_only=True)
class Game14BCtrlEncoderConfig(EncoderConfig):
    """Config for the proxy-DINO 14B condition encoder."""

    _target: type["Game14BCtrlEncoder"] = field(
        default_factory=lambda: Game14BCtrlEncoder
    )

    encoder: WanVAEEncoderConfig = field(default_factory=WanVAEEncoderConfig)
    """Streaming Wan VAE encoder; pin its checkpoint to the decoder's."""


class Game14BCtrlEncoder(StreamingVideoEncoder[Game14BCtrlEncoderCache]):
    """Assemble the per-chunk I2V + proxy-DINO control payload."""

    encoder: WanVAEEncoder

    def __init__(self, config: Game14BCtrlEncoderConfig) -> None:
        super().__init__(config)
        self.config: Game14BCtrlEncoderConfig = config
        self.encoder = config.encoder.setup()
        # The cover latent depends only on the anchor frame, which a session
        # keeps across restarts; compared by identity, not by value.
        self._cover_cache: tuple[Tensor, Tensor] | None = None

    def initialize_autoregressive_cache(self) -> Game14BCtrlEncoderCache:
        return Game14BCtrlEncoderCache(
            vae=self.encoder.initialize_autoregressive_cache(),
        )

    @torch.no_grad()
    def encode_cover(self, image: Tensor) -> Tensor:
        """Latent frame 0: the cover encoded alone through a fresh VAE cache."""
        if image.shape[-4] != 1:
            raise ValueError(
                f"first frame must have T=1, got shape {tuple(image.shape)}"
            )
        cached = self._cover_cache
        if cached is not None and cached[0] is image:
            return cached[1]
        latent = self.encoder(
            image,
            autoregressive_index=0,
            cache=self.encoder.initialize_autoregressive_cache(),
        )
        self._cover_cache = (image, latent)
        return latent

    @torch.no_grad()
    def forward(
        self,
        input: Game14BCtrlInput,
        autoregressive_index: int = 0,
        cache: Game14BCtrlEncoderCache | None = None,
    ) -> I2VProxyDinoCtrl:
        """Build this chunk's condition latent and pair it with the control."""
        assert cache is not None, "Game14BCtrlEncoder requires a per-rollout cache."
        control = input.control
        len_t = int(control.shape[-4])

        if autoregressive_index == 0:
            if input.image is None:
                raise ValueError(
                    "Game14BCtrlEncoder needs the cover pixels at AR step 0; the "
                    "packed first frame is encoded there and nowhere else."
                )
            cache.cover_latent = self.encode_cover(input.image)
            pixels = input.image
        elif cache.grey_tail is None or cache.cover_latent is None:
            raise ValueError(
                f"Game14BCtrlEncoder saw AR step {autoregressive_index} before AR "
                "step 0; the rollout must start at 0 so the packed first frame "
                "is encoded."
            )

        else:
            sp = self.spatial_compression_ratio
            pixels = cache.grey_tail.new_empty(
                *cache.grey_tail.shape[:-4],
                1,
                3,
                cache.grey_tail.shape[-2] * sp,
                cache.grey_tail.shape[-1] * sp,
                dtype=torch.float32,
            )
        decoded_frames = self.encoder.get_input_temporal_size(
            autoregressive_index, len_t
        )
        zeros = pixels.new_zeros(*pixels.shape[:-4], decoded_frames, *pixels.shape[-3:])
        if autoregressive_index == 0:
            zeros[..., :1, :, :, :] = pixels
        condition = self.encoder(
            zeros, autoregressive_index=autoregressive_index, cache=cache.vae
        )
        cache.grey_tail = condition
        mask = condition.new_zeros(
            *condition.shape[:-3], _MASK_TAG_CHANNELS, *condition.shape[-2:]
        )
        if autoregressive_index == 0:
            # Latent frame 0 of the rollout is the cover, not the tail's own
            # frame 0, and it is the only frame the mask marks.
            condition = torch.cat(
                [
                    cache.cover_latent.to(dtype=condition.dtype),
                    condition[..., 1:, :, :, :],
                ],
                dim=-4,
            )
            mask[..., 0, :, :, :] = 1.0
        return I2VProxyDinoCtrl(
            i2v=I2VCtrl(latent=condition, mask=mask),
            control=control.to(device=condition.device, dtype=condition.dtype),
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
    "Game14BCtrlEncoder",
    "Game14BCtrlEncoderCache",
    "Game14BCtrlEncoderConfig",
    "Game14BCtrlInput",
]
