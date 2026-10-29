# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the eden-local numerics: scheduler, fp32 AdaLN, kernel switches."""

from __future__ import annotations

import copy
from typing import cast

import pytest
import torch
import torch.nn.functional as F
from eden.impl.acceleration import SingleGPUAttentionOp
from eden.impl.numerics import EdenWanDiTNetwork, FP32AdaLNBlock, FP32AdaLNHead
from eden.impl.scheduler import EdenFlowMatchScheduler, EdenFlowMatchSchedulerConfig
from torch import nn

from flashdreams.recipes.wan.transformer.impl.modules import Block, Head

pytestmark = pytest.mark.ci_cpu


def _scheduler(**overrides) -> EdenFlowMatchScheduler:
    config = EdenFlowMatchSchedulerConfig(
        num_inference_steps=4,
        denoising_timesteps=[1000, 750, 500, 250],
        warp_denoising_step=True,
        shift=5.0,
        sigma_max=1.0,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=1000,
        timestep_dtype=torch.float32,
        network_timestep_dtype=torch.float32,
        **overrides,
    )
    return config.setup()


def test_network_sees_exact_fractional_timesteps() -> None:
    """937.5 / 833.33 / 625 reach the network unrounded (bf16 would give 936/832/624)."""
    scheduler = _scheduler()
    seen: list[torch.Tensor] = []

    def predict_flow(
        noisy_latent: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        seen.append(timestep)
        return torch.zeros_like(noisy_latent)

    scheduler.sample(torch.zeros(2, 4, dtype=torch.bfloat16), predict_flow)
    assert [t.dtype for t in seen] == [torch.float32] * 4
    assert [round(t.item(), 3) for t in seen] == [1000.0, 937.5, 833.333, 625.0]
    assert scheduler.network_timestep_dtype(torch.bfloat16) is torch.float32


def test_default_network_dtype_follows_the_latent() -> None:
    scheduler = _scheduler()
    scheduler.config.network_timestep_dtype = None
    assert scheduler.network_timestep_dtype(torch.bfloat16) is torch.bfloat16


def test_renoise_uses_noise_fn_and_bf16_sigma() -> None:
    """x0 = x - sigma v and the re-noise lerp run in the latent dtype."""
    scheduler = _scheduler()
    flow = torch.full((3,), 0.25, dtype=torch.bfloat16)
    noise = torch.full((3,), -1.0, dtype=torch.bfloat16)
    calls: list[tuple[int, ...]] = []

    def noise_fn(like: torch.Tensor) -> torch.Tensor:
        calls.append(tuple(like.shape))
        return noise

    def predict_flow(
        noisy_latent: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        del noisy_latent, timestep
        return flow

    out = scheduler.sample(
        torch.ones(3, dtype=torch.bfloat16), predict_flow, noise_fn=noise_fn
    )
    assert calls == [(3,)] * 3
    # Replay the loop by hand in bf16.
    noisy = torch.ones(3, dtype=torch.bfloat16)
    clean = torch.empty(0)
    for i, sigma in enumerate(scheduler.denoising_sigmas.tolist()):
        s = torch.tensor(sigma, dtype=torch.bfloat16)
        if i:
            noisy = ((1.0 - s) * clean + s * noise).to(torch.bfloat16)
        clean = noisy - s * flow
    assert torch.equal(out, clean)


def test_lingbot_reference_sampling_is_rejected() -> None:
    with pytest.raises(ValueError, match="reference_sampling"):
        _scheduler(reference_sampling=True)


class _SDPA(nn.Module):
    """CPU stand-in for ``ContextParallelAttention`` (``bshd`` layout)."""

    def set_context_parallel_group(self, cp_group) -> None:
        del cp_group

    def forward(self, q, k, v):
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        return out.transpose(1, 2)


def _tiny_block(dtype: torch.dtype) -> Block:
    torch.manual_seed(0)
    block = Block(dim=64, ffn_dim=128, num_heads=2, cross_attn_norm=True)
    # CPU stand-ins for the CUDA attention ops (typed as the core op).
    setattr(block.self_attn, "attn_op", _SDPA())
    setattr(block.cross_attn, "attn_op", _SDPA())
    with torch.no_grad():
        block.modulation.normal_(0, 0.5)
        cast(nn.LayerNorm, block.norm3).weight.normal_(1, 0.1)
    block.update_parameters_after_loading_checkpoint()
    return block.to(dtype)


def _run(block: Block, x, e, context) -> torch.Tensor:
    cache = block.initialize_cache(
        chunk_size=x.shape[1], window_size=x.shape[1], sink_size=0, context_text=context
    )
    cache.self_attn.before_update(0)
    return block(x, e, cache, rope_freqs=None)


def test_fp32_adaln_block_equals_the_stock_block_in_fp32() -> None:
    """Same parameters, same maths: identical when the stream is already fp32."""
    stock = _tiny_block(torch.float32)
    fp32 = copy.deepcopy(stock)
    fp32.__class__ = FP32AdaLNBlock
    x = torch.randn(1, 8, 64)
    e = torch.randn(1, 6, 64)
    context = torch.randn(1, 5, 64)
    torch.testing.assert_close(_run(fp32, x, e, context), _run(stock, x, e, context))


def test_fp32_adaln_block_is_closer_to_fp64_in_bf16() -> None:
    """In bf16 the Diffusers precision path tracks the fp64 result more closely."""
    reference = _tiny_block(torch.float64)
    x = torch.randn(1, 8, 64, dtype=torch.float64) * 4
    e = torch.randn(1, 6, 64, dtype=torch.float64)
    context = torch.randn(1, 5, 64, dtype=torch.float64)
    truth = _run(reference, x, e, context)

    stock = _tiny_block(torch.bfloat16)
    fp32 = copy.deepcopy(stock)
    fp32.__class__ = FP32AdaLNBlock
    args = (x.bfloat16(), e.bfloat16(), context.bfloat16())
    err_stock = (_run(stock, *args).double() - truth).abs().mean()
    err_fp32 = (_run(fp32, *args).double() - truth).abs().mean()
    assert err_fp32 < err_stock


def test_fp32_adaln_head_equals_the_stock_head_in_fp32() -> None:
    torch.manual_seed(0)
    head = Head(64, 4, (1, 2, 2))
    head.update_parameters_after_loading_checkpoint()
    fp32 = copy.deepcopy(head)
    fp32.__class__ = FP32AdaLNHead
    x, e = torch.randn(1, 8, 64), torch.randn(1, 1, 64)
    torch.testing.assert_close(fp32(x, e), head(x, e))


def _tiny_network(**flags) -> EdenWanDiTNetwork:
    from eden.impl.game5b.network import Game5BDiTNetworkConfig

    config = Game5BDiTNetworkConfig(
        num_layers=2,
        dim=64,
        ffn_dim=128,
        num_heads=2,
        text_dim=32,
        freq_dim=16,
        control_dim=16,
        in_dim=48 + 16,
        **flags,
    )
    return config.setup()


def test_default_flags_build_the_stock_network() -> None:
    network = _tiny_network()
    assert all(type(block) is Block for block in network.blocks)
    assert type(network.head) is Head
    block = cast(Block, network.blocks[0])
    assert type(block.self_attn.attn_op).__name__ == "ContextParallelAttention"


def test_fp32_adaln_flag_reclasses_blocks_and_head_only() -> None:
    stock = _tiny_network()
    network = _tiny_network(fp32_adaln=True)
    assert all(type(block) is FP32AdaLNBlock for block in network.blocks)
    assert type(network.head) is FP32AdaLNHead
    # Checkpoint keys are unchanged.
    assert list(network.state_dict()) == list(stock.state_dict())


def test_sage_fp8_flag_swaps_only_the_self_attention_op() -> None:
    network = _tiny_network(self_attention_backend="sage_fp8")
    block = cast(Block, network.blocks[0])
    op = cast(SingleGPUAttentionOp, block.self_attn.attn_op)
    assert type(op) is SingleGPUAttentionOp
    assert type(block.cross_attn.attn_op).__name__ == "ContextParallelAttention"
    with pytest.raises(NotImplementedError):
        op.set_context_parallel_group(object())
