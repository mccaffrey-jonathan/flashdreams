# SPDX-FileCopyrightText: Copyright (c) 2026 Vast AI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for FlashDreams v0.2.0: the EdenLive fork added an ``fp32_adaln``
# switch to flashdreams/recipes/wan/transformer/impl/{modules,network}.py; the
# same arithmetic lives here as subclasses so the core Wan blocks stay unchanged.

"""Diffusers-precision AdaLN for the EdenLive Wan DiTs.

The EdenLive checkpoints were trained and evaluated through Diffusers'
``WanTransformerBlock``, which keeps the AdaLN modulation, every LayerNorm and
the self-attention / FFN residual adds in fp32 while the matmuls run in bf16.
FlashDreams' :class:`Block` runs all of it in the stream dtype. The classes here
reproduce the Diffusers precision path; parameter names and shapes are
unchanged, so the same checkpoint loads into either.

:class:`EdenWanDiTNetwork` switches it on per network with
``config.fp32_adaln`` by re-classing the blocks, the head and (14B) the CLIP
projection's LayerNorms after the base constructor builds them.
"""

from __future__ import annotations

from torch import Tensor, nn

from flashdreams.recipes.wan.transformer.impl.modules import Block, BlockCache, Head
from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetwork,
    WanDiTNetworkConfig,
)

from eden.impl.acceleration import replace_large_linears_with_fp8, replace_self_attention_ops


def _layer_norm_fp32(norm: nn.Module, x: Tensor) -> Tensor:
    """Apply ``norm`` to ``x`` in fp32 (Diffusers ``FP32LayerNorm``).

    ``nn.Identity`` (no cross-attn norm) passes an fp32 copy through, so the
    caller's ``.type_as`` round-trips it unchanged.
    """
    if not isinstance(norm, nn.LayerNorm):
        return norm(x.float())
    return nn.functional.layer_norm(
        x.float(),
        norm.normalized_shape,
        None if norm.weight is None else norm.weight.float(),
        None if norm.bias is None else norm.bias.float(),
        norm.eps,
    )


class FP32LayerNorm(nn.LayerNorm):
    """``LayerNorm`` computed in fp32 and cast back, as Diffusers' ``FP32LayerNorm``."""

    def forward(self, x: Tensor) -> Tensor:
        return _layer_norm_fp32(self, x).to(x.dtype)


class FP32AdaLNBlock(Block):
    """:class:`Block` on the Diffusers ``WanTransformerBlock`` precision path.

    Modulation is ``modulation + e.float()``; each LayerNorm runs on an fp32
    copy of the residual stream; the shift/scale affine and the self-attention
    and FFN residual adds happen in fp32 before casting back. The
    cross-attention residual is added in the stream dtype, as upstream.
    """

    def forward(
        self,
        x: Tensor,
        e: Tensor,
        cache: BlockCache,
        rope_freqs: Tensor,
    ) -> Tensor:
        assert self._parameters_updated_after_loading_checkpoint, (
            "We expect to have called update_parameters_after_loading_checkpoint() "
            "before running the forward pass"
        )
        e_chunks = [c.squeeze(-2) for c in (self.modulation + e.float()).chunk(6, dim=-2)]

        y = (_layer_norm_fp32(self.norm1, x) * (1 + e_chunks[1]) + e_chunks[0]).type_as(x)
        y = self.self_attn(y, rope_freqs=rope_freqs, kv_cache=cache.self_attn)
        x = (x.float() + y * e_chunks[2]).type_as(x)

        x = x + self.cross_attn(
            _layer_norm_fp32(self.norm3, x).type_as(x),
            kv_cache=cache.cross_attn,
        )
        y = (_layer_norm_fp32(self.norm2, x) * (1 + e_chunks[4]) + e_chunks[3]).type_as(x)
        y = self.ffn(y)
        x = (x.float() + y.float() * e_chunks[5]).type_as(x)
        return x


class FP32AdaLNHead(Head):
    """:class:`Head` with the Diffusers ``norm_out`` precision.

    The shift/scale sum stays in the parameter dtype; the LayerNorm and the
    affine run in fp32.
    """

    def forward(self, x: Tensor, e: Tensor) -> Tensor:
        assert self._parameters_updated_after_loading_checkpoint, (
            "We expect to have called update_parameters_after_loading_checkpoint() "
            "before running the forward pass"
        )
        e_chunks = [c.squeeze(-2) for c in (self.modulation + e).chunk(2, dim=-2)]
        x = (self.norm(x.float()) * (1 + e_chunks[1]) + e_chunks[0]).type_as(x)
        return self.head(x)


class EdenWanDiTNetwork(WanDiTNetwork):
    """:class:`WanDiTNetwork` with opt-in precision and kernel switches.

    Reads from the config (every default is the stock network bit for bit):

    * ``fp32_adaln`` -- Diffusers-precision AdaLN (this module);
    * ``linear_backend`` / ``self_attention_backend`` -- low-precision kernels
      (:mod:`eden.impl.acceleration`).
    """

    def __init__(self, config: WanDiTNetworkConfig) -> None:
        super().__init__(config)
        self.fp32_adaln = bool(getattr(config, "fp32_adaln", False))
        self.linear_backend = getattr(config, "linear_backend", "torch")
        self.self_attention_backend = getattr(config, "self_attention_backend", "cudnn")
        replace_self_attention_ops(self.blocks, self.self_attention_backend)
        if not self.fp32_adaln:
            return
        # Re-class rather than rebuild: the subclasses add no state, so the
        # parameters (and the checkpoint keys) are exactly the base ones.
        for block in self.blocks:
            assert type(block) is Block, f"unexpected block type {type(block).__name__}"
            block.__class__ = FP32AdaLNBlock
        assert type(self.head) is Head
        self.head.__class__ = FP32AdaLNHead
        img_emb = getattr(self, "img_emb", None)
        if img_emb is not None:
            for module in img_emb.modules():
                if type(module) is nn.LayerNorm:
                    module.__class__ = FP32LayerNorm

    def update_parameters_after_loading_checkpoint(self) -> None:
        """Fuse load-time ops, then derive the FP8 block projections if asked.

        Runs after the checkpoint load (the FP8 weights are quantized from the
        real values) and before ``torch.compile`` / CUDA-graph capture.
        """
        if self._parameters_updated_after_loading_checkpoint:
            return
        super().update_parameters_after_loading_checkpoint()
        if self.linear_backend == "rowwise_fp8":
            replace_large_linears_with_fp8(self.blocks)
        elif self.linear_backend != "torch":
            raise ValueError(f"unknown linear_backend {self.linear_backend!r}")


__all__ = [
    "EdenWanDiTNetwork",
    "FP32AdaLNBlock",
    "FP32AdaLNHead",
    "FP32LayerNorm",
]
