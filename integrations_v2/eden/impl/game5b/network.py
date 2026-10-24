# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan 2.2 TI2V-5B DiT widened for the proxy-DINO channel-concat control.

The 5B recipe appends the 1024-channel proxy-DINO control to the model's
own 48 latent channels, giving a ``[3072, 1072, 1, 2, 2]`` patch embedding
(confirmed against the checkpoint's ``.metadata``). Two things follow from that
shape, and both are why this model needs no network subclass of its own:

* **No condition RMSNorm.** The training recipe built this run with
  ``enable_wan_channel_concat(..., condition_rmsnorm=False)``, so the patch
  embedding is a plain ``nn.Conv3d`` and the checkpoint carries no
  ``patch_embedding.condition_norm.weight``. The 14B AR variant is the opposite
  case and needs its own network class to hold that fp32 affine norm.
* **No CLIP branch.** Wan 2.2 TI2V-5B has no image cross-attention
  (``cross_attn_enable_img=False``), so the first frame is conditioned through
  the clean-latent stamp rather than CLIP features.

Channel order is ``[noisy 48 | proxy-DINO 1024]``. FlashDreams applies the
``conv3d`` patch embedding as an ``F.linear`` over already-patchified tokens,
whose layout is ``(c kt kh kw)`` with the channel axis outermost, so appending
the patchified control to the patchified latent on the last axis is the same
tensor the reference's pre-patchify channel concat produces.

The one thing the stock network does not provide is
:meth:`Game5BDiTNetwork.replace_text_embeddings`, the per-chunk dynamic-prompt
hook, so the subclass here exists to add it and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetwork,
    WanDiTNetworkCache,
    WanDiTNetworkTI2V5BConfig,
)


class Game5BDiTNetwork(WanDiTNetwork):
    """TI2V-5B DiT with the per-chunk prompt hook the dynamic path needs.

    The widened patch embedding is an ordinary ``nn.Conv3d`` and every other
    module is unchanged from the base network, so the only addition is
    :meth:`replace_text_embeddings`.
    """

    def replace_text_embeddings(
        self,
        cache: WanDiTNetworkCache,
        text_embeddings: Tensor,
    ) -> None:
        """Swap the cached cross-attention text K/V for every block.

        Interface for per-chunk dynamic prompts. The Game2V rollout encodes one
        static prompt at startup and releases the text encoder; a rollout that
        wants per-chunk prompts reloads it once through
        ``Game5BInferencePipeline.encode_text`` and passes the embeddings here.

        Only the static cross-attention context changes; the self-attention KV
        cache is untouched, so the rollout horizon survives the swap.

        The write goes through :meth:`BlockKVCache.overwrite_kv_` rather than
        rebinding ``cross_attn.text`` to a freshly built cache. The transformer
        forward reads the cross-attention K/V out of this cache while it runs
        inside a captured CUDA graph, and a captured kernel holds the *storage
        address* it was traced against. Rebinding the attribute hands the next
        eager forward a new allocation while the replayed graph keeps reading
        the old one -- which, once the old tensor is freed and the allocator
        reuses it, is whatever now occupies that memory. The dynamic-prompt
        swap is the only thing that mutates this cache mid-rollout, and it
        fires exactly when the game raises an event, so the corruption shows up
        as "the picture breaks the moment the player moves" with compile on.
        Writing in place keeps the addresses stable, so the graph replays
        correctly and behaves the same with or without compile / CUDA graphs.
        """
        assert text_embeddings.shape[-2] == self.text_len, (
            f"text embeddings must have {self.text_len} tokens, got "
            f"{text_embeddings.shape[-2]}"
        )
        context_text = self.text_embedding(text_embeddings)
        for block, block_cache in zip(self.blocks, cache.block_caches):
            fresh = block.cross_attn.compute_kv(context_text)
            block_cache.cross_attn.text.overwrite_kv_(*fresh.clone_kv())

# The 5B VAE's latent width; the noisy latent occupies these leading channels.
GAME5B_LATENT_CHANNELS = 48
# Projected proxy-DINO feature width, appended after the latent channels.
GAME5B_CONTROL_DIM = 1024


@dataclass
class Game5BDiTNetworkConfig(WanDiTNetworkTI2V5BConfig):
    """Config for the proxy-DINO 5B network.

    ``in_dim`` covers the full channel-concat layout
    ``[noisy 48 | proxy-DINO 1024]``; the checkpoint's
    ``patch_embedding.weight`` is ``[3072, 1072, 1, 2, 2]``, so the convolution
    must be constructed at this width before loading.

    The network is :class:`Game5BDiTNetwork`, which is the stock Wan DiT plus
    the dynamic-prompt hook; the widened patch embedding is an ordinary
    ``nn.Conv3d`` and nothing else about the network changed.
    """

    _target: type[Game5BDiTNetwork] = Game5BDiTNetwork

    in_dim: int = GAME5B_LATENT_CHANNELS + GAME5B_CONTROL_DIM
    control_dim: int = GAME5B_CONTROL_DIM
    patch_embedding_type: str = "conv3d"
    # 24 heads do not shard evenly across 8 ranks, so token sharding rather
    # than head sharding is what covers an 8-GPU node here.
    cp_method: str = "ring"

    def __post_init__(self) -> None:
        if self.control_dim <= 0:
            raise ValueError("control_dim must be positive")
        expected = GAME5B_LATENT_CHANNELS + self.control_dim
        if self.in_dim != expected:
            raise ValueError(
                f"in_dim={self.in_dim} does not match the channel-concat layout "
                f"[noisy {GAME5B_LATENT_CHANNELS} | control {self.control_dim}] "
                f"= {expected}"
            )


__all__ = [
    "GAME5B_CONTROL_DIM",
    "GAME5B_LATENT_CHANNELS",
    "Game5BDiTNetwork",
    "Game5BDiTNetworkConfig",
]
