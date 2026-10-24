# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-precision kernels for the EdenLive DiTs on a single GeForce GPU.

Two independent switches, both off by default:

``linear_backend="rowwise_fp8"``
    Every large block ``nn.Linear`` (self/cross-attention projections and the
    FFN, ~98% of the 5B's parameters) becomes a dynamic row-wise FP8 E4M3
    projection: per-output-channel weight scales, per-token activation scales,
    bf16 output. Same recipe as the LingBot RTX 5090 ``perf`` preset; on
    Blackwell GeForce the GEMM runs through the in-tree Triton kernel.

``self_attention_backend``
    ``"cudnn"`` (default) keeps Wan's SDPA. ``"sage_fp8"`` swaps the
    self-attention op for the in-tree Sage-style kernel
    (:func:`~flashdreams.accelerated.multi_head_attention.triton.scaled_fp8_attention.scaled_fp8_attention`:
    INT8 Q/K per token block, FP8 V, FP32 softmax, K mean-centred).
    ``"sage2"`` calls the external SageAttention 2 package when it is installed
    (the kernel the ComfyUI/LightX2V reference uses).

The attention swap replaces only the ``attn_op`` the Wan ``SelfAttention``
calls *after* it has rotated Q and the cached K, so the EdenLive cache-slot RoPE
(K cached unrotated, rotated on read) is untouched. Both low-precision attention
ops are single-GPU only.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from flashdreams.accelerated.quantization.linear import (
    QuantizedNonPersistentLinear,
    WeightGranularity,
)
from flashdreams.accelerated.quantization.quantizer import Granularity, quantize

LinearBackend = Literal["torch", "rowwise_fp8"]
SelfAttentionBackend = Literal["cudnn", "sage_fp8", "sage2"]


def _rowwise_fp8_linear(projection: QuantizedNonPersistentLinear, x: Tensor) -> Tensor:
    """Apply ``projection`` with compile-fusible row-wise FP8 quantization."""
    if x.numel() == 0:
        return projection(x, Granularity.SLICE, out_dtype=x.dtype)
    quantized, scale = quantize(x, projection.dtype, Granularity.SLICE, axis=-1, use_triton=False)
    return projection(quantized, scale, out_dtype=x.dtype)


class DynamicFP8Linear(nn.Module):
    """``nn.Linear`` replacement with row-wise dynamic FP8 activations and weights."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.projection = QuantizedNonPersistentLinear(
            linear.weight,
            linear.bias,
            WeightGranularity.PER_OUT_CHANNEL,
            torch.float8_e4m3fn,
        )

    def forward(self, x: Tensor) -> Tensor:
        return _rowwise_fp8_linear(self.projection, x)


def replace_large_linears_with_fp8(module: nn.Module, min_features: int = 1024) -> int:
    """Swap aligned large ``nn.Linear`` descendants for :class:`DynamicFP8Linear`.

    Returns the number of layers replaced.
    """
    replaced = 0
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if type(child) is not nn.Linear:
                continue
            if (
                child.in_features < min_features
                or child.out_features < min_features
                or child.in_features % 16
                or child.out_features % 16
            ):
                continue
            setattr(parent, name, DynamicFP8Linear(child))
            replaced += 1
    return replaced


@torch.library.custom_op("eden::sage_fp8_attention", mutates_args=())
def _sage_fp8_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    from flashdreams.accelerated.multi_head_attention.triton.scaled_fp8_attention import (
        scaled_fp8_attention,
    )

    return scaled_fp8_attention(
        query,
        key,
        value,
        use_tma=True,
        qk_quantization_blocks=(128, 64),
        use_fp16_pv=True,
    )


@_sage_fp8_attention.register_fake
def _sage_fp8_attention_fake(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    return torch.empty_like(query)


@torch.library.custom_op("eden::sage2_attention", mutates_args=())
def _sage2_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    from sageattention import sageattn

    return sageattn(query, key, value, tensor_layout="NHD", is_causal=False)


@_sage2_attention.register_fake
def _sage2_attention_fake(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    return torch.empty_like(query)


class SingleGPUAttentionOp(nn.Module):
    """Drop-in for ``ContextParallelAttention`` (``bshd``) running a fused kernel."""

    def __init__(self, backend: Literal["sage_fp8", "sage2"]) -> None:
        super().__init__()
        self.backend = backend
        self._op = _sage_fp8_attention if backend == "sage_fp8" else _sage2_attention
        if backend == "sage2":
            try:
                import sageattention  # noqa: F401
            except ImportError as error:  # pragma: no cover - optional dependency
                raise ImportError(
                    "self_attention_backend='sage2' needs the SageAttention 2 "
                    "package (pip install sageattention, or a prebuilt wheel)"
                ) from error

    def set_context_parallel_group(self, cp_group) -> None:
        if cp_group is not None:
            raise NotImplementedError(
                f"self_attention_backend={self.backend!r} does not support context parallelism"
            )

    def is_context_parallel_enabled(self) -> bool:
        return False

    def context_parallel_size(self) -> int:
        return 1

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return self._op(query.contiguous(), key.contiguous(), value.contiguous())


def replace_self_attention_ops(blocks: nn.ModuleList, backend: SelfAttentionBackend) -> None:
    """Point every block's self-attention at ``backend``."""
    if backend == "cudnn":
        return
    for block in blocks:
        block.self_attn.attn_op = SingleGPUAttentionOp(backend)


__all__ = [
    "DynamicFP8Linear",
    "LinearBackend",
    "SelfAttentionBackend",
    "SingleGPUAttentionOp",
    "replace_large_linears_with_fp8",
    "replace_self_attention_ops",
]
