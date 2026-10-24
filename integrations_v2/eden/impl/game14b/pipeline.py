# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming pipeline for the proxy-DINO AR 14B Game2V model.

Differences from the base Wan I2V pipeline:

* The first-frame condition is built per chunk, never for the whole rollout:
  AR step 0 hands the cover to the encoder, which encodes it alone and splices
  it onto a grey condition stream whose global time advances every chunk. See
  :mod:`eden.impl.game14b.encoder`.
* ``generate`` takes the already-encoded proxy-DINO features for the chunk.
  Computing them is the caller's job precisely so the cost can be overlapped
  with the previous chunk's denoising.
* The CFG negative prompt is the T2V string with a leading ``镜头晃动，``.
  See :data:`GAME14B_NEGATIVE_PROMPT`.
* ``encode_text`` keeps the text encoder resident and
  ``replace_text_embeddings`` swaps the resulting embeddings into the
  transformer's cross-attention caches, which is the per-chunk prompt path.
  The static prompt is encoded once at startup and the text encoder is then
  released, so a rollout that never asks for a per-chunk prompt pays nothing.

The one-shot encoders' outputs (UMT5 text, CLIP vision) and the cover latent
are reused across rollouts. The Game2V session re-runs
``initialize_cache`` on every restart, and reloading both encoders from disk
plus re-encoding a prompt that has not changed is the whole of the stall a
restart shows. A prompt or anchor frame that actually changes recomputes only
the part that changed; the grey stream starts with a fresh VAE cache on each rollout.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger
from torch import Tensor

from flashdreams.infra.encoder.image.clip import CLIPImageEncoderConfig
from flashdreams.recipes.wan.pipeline import (
    WanInferencePipeline,
    WanInferencePipelineCache,
    WanInferencePipelineConfig,
)
from .encoder import Game14BCtrlEncoderConfig, Game14BCtrlInput

GAME14B_NEGATIVE_PROMPT = (
    "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
    "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
"""The 14B CFG negative prompt: the T2V ``NEGATIVE_PROMPT`` with a leading
``镜头晃动，``; the shared constant is left as is because the
other Wan pipelines use it."""


@dataclass(kw_only=True)
class Game14BInferencePipelineConfig(WanInferencePipelineConfig):
    """Config for the proxy-DINO AR 14B pipeline."""

    _target: type["Game14BInferencePipeline"] = field(
        default_factory=lambda: Game14BInferencePipeline
    )

    encoder: Game14BCtrlEncoderConfig = field(default_factory=Game14BCtrlEncoderConfig)
    """Condition encoder: first-frame latent slices + proxy-DINO control."""

    image_encoder: CLIPImageEncoderConfig | None = field(
        default_factory=CLIPImageEncoderConfig
    )
    """Wan 2.1 I2V ships a CLIP vision tower; the recipe keeps it enabled."""


class Game14BInferencePipeline(WanInferencePipeline):
    """Streaming I2V pipeline with proxy-DINO channel-concat control."""

    config: Game14BInferencePipelineConfig

    def __init__(self, config: Game14BInferencePipelineConfig) -> None:
        # The base constructor builds every component inline and returns only
        # when all of them exist, so a multi-minute weight load looks like a
        # hang. Time each piece so the startup log shows what is in progress.
        started = time.perf_counter()
        logger.info("Game2V 14B: building the streaming VAE encoder")
        super().__init__(config)
        logger.info(
            "Game2V 14B: pipeline components ready in {:.1f}s "
            "(VAE encoder, decoder, DiT, text and image encoders)",
            time.perf_counter() - started,
        )
        # The prompt and anchor frame are constant across a session's
        # restarts, so the one-shot encoder outputs are kept and reused
        # instead of reloading the encoders on every ``initialize_cache``.
        self._text_cache: tuple[tuple[str, ...], Tensor] | None = None
        self._negative_text_cache: Tensor | None = None
        self._image_cache: tuple[Tensor, Tensor | None] | None = None
        # Set while the base ``initialize_cache`` runs, so the encoder
        # overrides below know which inputs the outputs belong to.
        self._pending_text: tuple[str, ...] | None = None
        self._pending_image: Tensor | None = None

    @staticmethod
    def _negative_prompt() -> str:
        """The 14B negative prompt (see :data:`GAME14B_NEGATIVE_PROMPT`)."""
        return GAME14B_NEGATIVE_PROMPT

    @torch.no_grad()
    def initialize_cache(
        self,
        text: list[str],
        image: Tensor | None = None,
        *,
        height: int | None = None,
        width: int | None = None,
        release_oneshot_encoders: bool = True,
    ) -> WanInferencePipelineCache:
        """Build the rollout cache.

        The first-frame condition is not built here: AR step 0 encodes the
        cover, and the grey tail is cached on the encoder per resolution.

        The text and CLIP embeddings are cached
        against their inputs. A restart that re-runs this method with the
        same prompt and anchor frame reuses them, so the UMT5 and CLIP
        encoders are loaded exactly once per process instead of once per
        rollout.
        """
        # Reuse before the base call: ``super().initialize_cache`` reloads the
        # released one-shot encoders and re-encodes unconditionally, which is
        # the restart stall this cache exists to avoid.
        if self.diffusion_model.config.max_latent_frames is not None:
            self.diffusion_model._rng = None
            self.diffusion_model._initial_video_noise = None
        text_embeddings = self._cached_text_embeddings(text)
        image_embeddings = (
            self._cached_image_embeddings(image) if image is not None else None
        )
        reusing = text_embeddings is not None and (
            image is None or image_embeddings is not None
        )
        # The first call loads UMT5-XXL and CLIP from disk; a restart with the
        # same inputs skips all of it. The two look identical from the outside otherwise.
        logger.info(
            "Game2V 14B: {} the rollout cache",
            "reusing cached encoder outputs for" if reusing else "building",
        )
        cache_started = time.perf_counter()
        cache = self._initialize_cache_from_embeddings(
            text_embeddings,
            image,
            image_embeddings,
            height=height,
            width=width,
        )
        if cache is None:
            # First time these inputs are seen: the base call loads the
            # one-shot encoders, encodes, and releases them. The wrappers
            # record what it computed so the next call skips all of it.
            self._pending_text = tuple(text)
            self._pending_image = image
            cache = self._initialize_cache_recording(
                text,
                image,
                height=height,
                width=width,
                release_oneshot_encoders=release_oneshot_encoders,
            )
            self._pending_text = None
            self._pending_image = None

        assert cache.image is not None, (
            "Game14BInferencePipeline is I2V-only; pass ``image=...`` to "
            "``initialize_cache``."
        )
        assert cache.encoder_cache is not None  # invariant: encoder configured
        logger.info(
            "Game2V 14B: rollout cache ready in {:.2f}s ({})",
            time.perf_counter() - cache_started,
            "encoder outputs reused" if reusing else "encoders loaded and released",
        )
        return cache

    def _cached_text_embeddings(self, text: list[str]) -> Tensor | None:
        """Return the embeddings encoded for ``text`` earlier, if any."""
        cached = self._text_cache
        if cached is None or cached[0] != tuple(text):
            return None
        return cached[1]

    def _cached_image_embeddings(self, image: Tensor) -> Tensor | None:
        """Return the CLIP embeddings encoded for ``image`` earlier, if any.

        Compared by storage identity, not by value: the session re-runs cache
        initialization with the same loaded anchor tensor, and a pixel-wise
        comparison would cost more than it saves.
        """
        cached = self._image_cache
        if cached is None or cached[0] is not image:
            return None
        return cached[1]

    def _initialize_cache_recording(
        self,
        text: list[str],
        image: Tensor | None,
        *,
        height: int | None,
        width: int | None,
        release_oneshot_encoders: bool,
    ) -> WanInferencePipelineCache:
        """Run the base cache init while capturing the encoder outputs.

        The base implementation calls the encoders directly and keeps nothing
        but the transformer cache, so the outputs are intercepted here by
        wrapping the encoders for the duration of the call.
        """
        text_encoder = self.text_encoder
        image_encoder = self.image_encoder

        class _Recording(torch.nn.Module):
            """Forwards one encoder call and keeps its output.

            An ``nn.Module`` rather than a plain object because the encoders
            are registered submodules: assigning anything else is rejected.
            """

            def __init__(self, inner: torch.nn.Module, record: Any) -> None:
                super().__init__()
                self.inner = inner
                self._record = record

            def forward(self, *args: Any, **kwargs: Any) -> Any:
                output = self.inner(*args, **kwargs)
                self._record(*args, output)
                return output

        try:
            if isinstance(text_encoder, torch.nn.Module):
                self.text_encoder = _Recording(
                    text_encoder, self._record_text_embeddings
                )
            if isinstance(image_encoder, torch.nn.Module):
                pending_image = self._pending_image

                def _record_image(_image: Tensor, embeddings: Tensor) -> None:
                    # The base squeezes the T=1 axis before encoding, so the
                    # tensor seen here is not the anchor frame. The pending
                    # frame is: it is the one the caller asked to encode.
                    if pending_image is not None:
                        self._image_cache = (pending_image, embeddings)

                self.image_encoder = _Recording(image_encoder, _record_image)
            return super().initialize_cache(
                text,
                image,
                height=height,
                width=width,
                release_oneshot_encoders=release_oneshot_encoders,
            )
        finally:
            # The base releases the one-shot encoders when asked, which drops
            # the wrapper. Only restore an encoder it actually kept; restoring
            # one it released would pin the weights in memory permanently.
            if self.text_encoder is not None:
                self.text_encoder = text_encoder
            if self.image_encoder is not None:
                self.image_encoder = image_encoder
            gc.collect()

    def _record_text_embeddings(self, text: list[str], embeddings: Tensor) -> None:
        """Keep embeddings the base call just computed, keyed by their prompt."""
        if self._pending_text is not None and tuple(text) == self._pending_text:
            self._text_cache = (self._pending_text, embeddings)
        elif text == [self._negative_prompt()]:
            self._negative_text_cache = embeddings

    def _initialize_cache_from_embeddings(
        self,
        text_embeddings: Tensor,
        image: Tensor | None,
        image_embeddings: Tensor | None,
        *,
        height: int | None,
        width: int | None,
    ) -> WanInferencePipelineCache | None:
        """Build a rollout cache without touching the one-shot encoders.

        Returns ``None`` when ``text_embeddings`` (or, for I2V, the CLIP
        embeddings) have not been computed yet, so the caller falls back to
        the base path that loads and runs the encoders.

        Mirrors the tail of ``WanInferencePipeline.initialize_cache`` once the
        embeddings are already known: derive the latent size from the anchor
        frame and hand the embeddings to the base pipeline cache builder.
        """
        if text_embeddings is None:
            return None
        if image is not None and image_embeddings is None:
            return None
        if image is not None:
            if self.decoder is None:
                raise RuntimeError(
                    "Game14BInferencePipeline requires a decoder to derive the "
                    "latent size from the anchor frame."
                )
            sp = self.decoder.spatial_compression_ratio
            pixel_h, pixel_w = int(image.shape[-2]), int(image.shape[-1])
            if pixel_h % sp != 0 or pixel_w % sp != 0:
                raise ValueError(
                    f"image pixel size ({pixel_h}, {pixel_w}) must be divisible "
                    f"by decoder.spatial_compression_ratio={sp}."
                )
            derived_h, derived_w = pixel_h // sp, pixel_w // sp
            if height is None:
                height = derived_h
            elif height != derived_h:
                raise ValueError(
                    f"height={height} does not match image latent height "
                    f"derived from pixels ({derived_h})."
                )
            if width is None:
                width = derived_w
            elif width != derived_w:
                raise ValueError(
                    f"width={width} does not match image latent width "
                    f"derived from pixels ({derived_w})."
                )
        if height is None or width is None:
            raise ValueError("T2V (image=None) requires explicit `height` and `width`.")

        guidance_scale = self._transformer_config.guidance_scale
        negative_text_embeddings = None
        if guidance_scale > 1.0:
            # CFG is off for this recipe, so this branch is never taken. It
            # stays for a config that turns guidance back on: the negative
            # prompt is constant, so it is encoded once and kept.
            negative_text_embeddings = self._negative_text_cache
            if negative_text_embeddings is None:
                self._ensure_oneshot_encoders_loaded()
                assert self.text_encoder is not None
                negative_text_embeddings = self.text_encoder([self._negative_prompt()])
                self._negative_text_cache = negative_text_embeddings

        parent = super(WanInferencePipeline, self).initialize_cache(
            transformer_context={
                "height": height,
                "width": width,
                "text_embeddings": text_embeddings,
                "negative_text_embeddings": negative_text_embeddings,
                "image_embeddings": image_embeddings,
            },
        )
        return WanInferencePipelineCache(
            transformer_cache=parent.transformer_cache,
            encoder_cache=parent.encoder_cache,
            decoder_cache=parent.decoder_cache,
            image=image,
        )

    @torch.no_grad()
    def generate(  # type: ignore[override]
        self,
        autoregressive_index: int,
        cache: WanInferencePipelineCache,
        input: Tensor,
    ) -> Tensor:
        """Generate one decoded chunk from precomputed proxy-DINO features.

        Args:
            autoregressive_index: AR step index, starting at 0.
            cache: Cache from :meth:`initialize_cache`.
            input: Proxy-DINO control for this chunk,
                ``[..., len_t, 1024, Hl, Wl]``. Already encoded, so this call
                never waits on DINO.

        Returns:
            Decoded video ``[*batch_shape, T, C, H, W]`` in ``[-1, 1]``.
        """
        assert cache.image is not None, "initialize_cache() must be called first"
        maximum = self.diffusion_model.config.max_latent_frames
        if maximum is not None:
            start = autoregressive_index * self.diffusion_model.transformer.config.len_t
            input = input[..., : maximum - start, :, :, :]

        # The packed first frame is encoded at AR 0 only (the cover alone, no
        # zero padding: the tail comes from the encoder's grey-tail cache).
        image = cache.image if autoregressive_index == 0 else None
        # Skip ``WanInferencePipeline.generate``: it would rebuild the input
        # from ``cache.image`` and drop the control payload.
        return super(WanInferencePipeline, self).generate(
            autoregressive_index=autoregressive_index,
            cache=cache,
            input=Game14BCtrlInput(image=image, control=input),
        )

    @torch.no_grad()
    def encode_text(self, text: list[str]) -> Tensor:
        """Encode ``text`` and keep the text encoder resident afterwards.

        The static rollout encodes once inside :meth:`initialize_cache` and
        then frees the encoder. A per-chunk prompt cannot pay that reload on
        every chunk, so the first call here loads it and no later call releases
        it. The caller decides which rank runs this; the weights land on
        ``self.device``.
        """
        self._ensure_oneshot_encoders_loaded()
        assert self.text_encoder is not None, "text_encoder is not set"
        embeddings = self.text_encoder(text)
        # The recording wrapper only exists while a cache is being built.
        self._text_cache = (tuple(text), embeddings)
        return embeddings

    @torch.no_grad()
    def replace_text_embeddings(
        self,
        cache: WanInferencePipelineCache,
        text_embeddings: Tensor,
    ) -> None:
        """Swap already-encoded text into the transformer's KV caches.

        Only the static cross-attention context changes; the self-attention KV
        cache is untouched, so the rollout horizon survives the swap. Every
        rank holds its own cache, so every rank runs this on the same
        embeddings. The text encoder is not touched here: encoding happens once,
        on the rank that owns it, and the embeddings are what travels.
        """
        transformer = self.diffusion_model.transformer
        network = getattr(transformer.network, "_orig_mod", transformer.network)
        network.replace_text_embeddings(
            cache.transformer_cache.network_cache, text_embeddings
        )


__all__ = [
    "Game14BInferencePipeline",
    "Game14BInferencePipelineConfig",
]
