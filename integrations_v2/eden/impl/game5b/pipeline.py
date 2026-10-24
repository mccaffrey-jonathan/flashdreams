# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming pipeline for the proxy-DINO 5B Game2V model.

Differences from the base Wan TI2V pipeline:

* ``generate`` takes the already-encoded proxy-DINO features for the chunk and
  pairs them with the TI2V pixel chunk the base path would have built on its
  own. Computing the features is the caller's job precisely so the cost can be
  overlapped with the previous chunk's denoising.
* ``encode_text`` keeps the text encoder resident and
  ``replace_text_embeddings`` swaps the resulting embeddings into the
  transformer's cross-attention caches, which is the per-chunk prompt path. The
  static prompt is encoded once at startup and the text encoder is then
  released, so a rollout that never asks for a per-chunk prompt pays nothing.
* The CFG negative prompt is the recipe's own (English) string rather than
  Eden's camera-model prompt. CFG is off at ``guidance_scale=1.0``, so it is
  only reachable by a config that turns guidance back on.

The one-shot text encoder's output is reused across rollouts. The Game2V session
re-runs ``initialize_cache`` on every restart, and reloading UMT5-XXL from disk
plus re-encoding a prompt that has not changed is the whole of the stall a
restart shows. There is no CLIP vision tower to reload: Wan 2.2 TI2V-5B has
none, and the first frame is conditioned through the clean-latent stamp
instead. The first-frame condition latent is likewise not cached here -- unlike
the 14B's whole-rollout condition clip, this model's is a single latent frame
the encoder produces inside AR step 0.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger
from torch import Tensor

from flashdreams.recipes.wan.pipeline import (
    WanInferencePipeline,
    WanInferencePipelineCache,
    WanInferencePipelineConfig,
)

from flashdreams.infra.encoder.text.umt5 import UMT5TextEncoderConfig

from .encoder import Game5BCtrlEncoderConfig, Game5BCtrlInput

# The model's negative prompt. Unlike Wan's stock Chinese string, the 5B was
# trained and validated against this English one.
NEGATIVE_PROMPT = (
    "Vibrant colors, overexposed, static, blurry details, subtitles, style, "
    "artwork, painting, image, still, overall grayish, worst quality, low "
    "quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn face, deformed, disfigured, deformed "
    "limbs, fused fingers, three legs"
)


@dataclass(kw_only=True)
class Game5BInferencePipelineConfig(WanInferencePipelineConfig):
    """Config for the proxy-DINO 5B pipeline."""

    _target: type["Game5BInferencePipeline"] = field(
        default_factory=lambda: Game5BInferencePipeline
    )

    encoder: Game5BCtrlEncoderConfig = field(default_factory=Game5BCtrlEncoderConfig)
    """Condition encoder: packed first-frame stamp + proxy-DINO control."""

    image_encoder: None = None
    """Wan 2.2 TI2V-5B ships no CLIP vision tower, so there is none to build.

    Pinned to ``None`` rather than left to the base default so a derived config
    cannot switch on a DiT cross-attention branch these weights do not have.
    """

    text_encoder: UMT5TextEncoderConfig | None = None
    """UMT5 text encoder. The directory must hold ``text_encoder/`` and
    ``tokenizer/`` side by side, which is the layout both ``from_pretrained``
    calls read. Set by the pipeline YAML; there is no built-in default path."""


class Game5BInferencePipeline(WanInferencePipeline):
    """Streaming TI2V pipeline with proxy-DINO channel-concat control."""

    config: Game5BInferencePipelineConfig

    def __init__(self, config: Game5BInferencePipelineConfig) -> None:
        # The base constructor builds every component inline and returns only
        # when all of them exist, so a multi-minute weight load looks like a
        # hang. Time each piece so the startup log shows what is in progress.
        started = time.perf_counter()
        logger.info("Game2V 5B: building the streaming VAE encoder")
        super().__init__(config)
        logger.info(
            "Game2V 5B: pipeline components ready in {:.1f}s "
            "(VAE encoder, decoder, DiT, text encoder)",
            time.perf_counter() - started,
        )
        # The prompt is constant across a session's restarts, so the one-shot
        # encoder output is kept and reused instead of reloading UMT5 on every
        # ``initialize_cache``.
        self._text_cache: tuple[tuple[str, ...], Tensor] | None = None
        self._negative_text_cache: Tensor | None = None
        # Set while the base ``initialize_cache`` runs, so the recording wrapper
        # below knows which inputs the outputs belong to.
        self._pending_text: tuple[str, ...] | None = None

    @staticmethod
    def _negative_prompt() -> str:
        """The recipe's own CFG negative prompt."""
        return NEGATIVE_PROMPT

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
        """Build the rollout cache, reusing the text embeddings when unchanged.

        The embeddings are cached against their prompt. A restart that re-runs
        this method with the same prompt reuses them, so UMT5 is loaded exactly
        once per process instead of once per rollout.
        """
        # Reuse before the base call: ``super().initialize_cache`` reloads the
        # released one-shot encoder and re-encodes unconditionally, which is the
        # restart stall this cache exists to avoid.
        if self.diffusion_model.config.max_latent_frames is not None:
            self.diffusion_model._rng = None
            self.diffusion_model._initial_video_noise = None
        text_embeddings = self._cached_text_embeddings(text)
        reusing = text_embeddings is not None
        logger.info(
            "Game2V 5B: {} the rollout cache",
            "reusing cached text embeddings for" if reusing else "building",
        )
        cache_started = time.perf_counter()
        cache = self._initialize_cache_from_embeddings(
            text_embeddings,
            image,
            height=height,
            width=width,
        )
        if cache is None:
            # First time this prompt is seen: the base call loads the one-shot
            # encoder, encodes, and releases it. The wrapper records what it
            # computed so the next call skips all of it.
            self._pending_text = tuple(text)
            try:
                cache = self._initialize_cache_recording(
                    text,
                    image,
                    height=height,
                    width=width,
                    release_oneshot_encoders=release_oneshot_encoders,
                )
            finally:
                self._pending_text = None

        assert cache.image is not None, (
            "Game5BInferencePipeline is TI2V-only; pass ``image=...`` to "
            "``initialize_cache``."
        )
        logger.info(
            "Game2V 5B: rollout cache ready in {:.2f}s ({})",
            time.perf_counter() - cache_started,
            "text embeddings reused" if reusing else "text encoder loaded and released",
        )
        return cache

    def _cached_text_embeddings(self, text: list[str]) -> Tensor | None:
        """Return the embeddings encoded for ``text`` earlier, if any."""
        cached = self._text_cache
        if cached is None or cached[0] != tuple(text):
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
        """Run the base cache init while capturing the text embeddings.

        The base implementation calls the encoder directly and keeps nothing but
        the transformer cache, so the output is intercepted here by wrapping the
        encoder for the duration of the call.

        The encoder is loaded here rather than left to the base call: after the
        first rollout released it, ``self.text_encoder`` is ``None`` and there
        would be nothing to wrap, so the embeddings the base call computes would
        go unrecorded and every restart with a new prompt would reload UMT5
        again. Loading first costs nothing -- the base call's own
        ``_ensure_oneshot_encoders_loaded`` is a no-op once the slot is filled.
        """
        self._ensure_oneshot_encoders_loaded()
        text_encoder = self.text_encoder

        class _Recording(torch.nn.Module):
            """Forwards one encoder call and keeps its output.

            An ``nn.Module`` rather than a plain object because the encoder is a
            registered submodule: assigning anything else is rejected.
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
            return super().initialize_cache(
                text,
                image,
                height=height,
                width=width,
                release_oneshot_encoders=release_oneshot_encoders,
            )
        finally:
            # The base releases the one-shot encoder when asked, which drops the
            # wrapper. Only restore an encoder it actually kept; restoring one it
            # released would pin the weights in memory permanently.
            if self.text_encoder is not None:
                self.text_encoder = text_encoder
            gc.collect()

    def _record_text_embeddings(self, text: list[str], embeddings: Tensor) -> None:
        """Keep embeddings the base call just computed, keyed by their prompt."""
        if self._pending_text is not None and tuple(text) == self._pending_text:
            self._text_cache = (self._pending_text, embeddings)
        elif text == [self._negative_prompt()]:
            self._negative_text_cache = embeddings

    def _initialize_cache_from_embeddings(
        self,
        text_embeddings: Tensor | None,
        image: Tensor | None,
        *,
        height: int | None,
        width: int | None,
    ) -> WanInferencePipelineCache | None:
        """Build a rollout cache without touching the one-shot text encoder.

        Returns ``None`` when ``text_embeddings`` have not been computed yet, so
        the caller falls back to the base path that loads and runs the encoder.

        Mirrors the tail of ``WanInferencePipeline.initialize_cache`` once the
        embeddings are already known: derive the latent size from the anchor
        frame and hand the embeddings to the base pipeline cache builder.
        """
        if text_embeddings is None:
            return None
        if image is not None:
            if self.decoder is None:
                raise RuntimeError(
                    "Game5BInferencePipeline requires a decoder to derive the "
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
            # CFG is off for this recipe, so this branch is never taken. It stays
            # for a config that turns guidance back on: the negative prompt is
            # constant, so it is encoded once and kept.
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
                "image_embeddings": None,
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

        # The packed first frame is encoded at AR 0 and stamped from there on,
        # so only that step needs pixels. Later steps would encode a zero chunk
        # whose latent the all-zero mask discards; the encoder skips it.
        pixels = cache.image if autoregressive_index == 0 else None
        # Skip ``WanInferencePipeline.generate``: it would rebuild the input from
        # ``cache.image`` and drop the control payload.
        return super(WanInferencePipeline, self).generate(
            autoregressive_index=autoregressive_index,
            cache=cache,
            input=Game5BCtrlInput(image=pixels, control=input),
        )

    @torch.no_grad()
    def encode_text(self, text: list[str]) -> Tensor:
        """Encode ``text`` and keep the text encoder resident afterwards.

        The static rollout encodes once inside :meth:`initialize_cache` and then
        frees the encoder. A per-chunk prompt cannot pay that reload on every
        chunk, so the first call here loads it and no later call releases it. The
        caller decides which rank runs this; the weights land on ``self.device``.
        """
        self._ensure_oneshot_encoders_loaded()
        assert self.text_encoder is not None, "text_encoder is not set"
        embeddings = self.text_encoder(text)
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
        cache is untouched, so the rollout horizon survives the swap. Every rank
        holds its own cache, so every rank runs this on the same embeddings. The
        text encoder is not touched here: encoding happens once, on the rank that
        owns it, and the embeddings are what travels.
        """
        transformer = self.diffusion_model.transformer
        network = getattr(transformer.network, "_orig_mod", transformer.network)
        network.replace_text_embeddings(
            cache.transformer_cache.network_cache, text_embeddings
        )


__all__ = [
    "NEGATIVE_PROMPT",
    "Game5BInferencePipeline",
    "Game5BInferencePipelineConfig",
]
