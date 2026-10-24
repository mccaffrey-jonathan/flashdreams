"""Regression coverage for bounded reference-compatible AR sampling."""

from types import SimpleNamespace

import torch

from flashdreams.core.attention.kvcache import BlockKVCache
from flashdreams.infra.diffusion.model.base import DiffusionModel
from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetworkCache,
    WanDiTNetworkConfig,
)


def test_video_noise_draw_preserves_channels_first_rng_and_cpu_storage():
    class Transformer(torch.nn.Module):
        config = SimpleNamespace(len_t=3)
        device = torch.device("cpu")
        dtype = torch.float32

        def unpatchify_and_maybe_gather_cp(self, value):
            return value

        def patchify_and_maybe_split_cp(self, value):
            return value

    model = DiffusionModel.__new__(DiffusionModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(seed=2026092901, max_latent_frames=7)
    model.transformer = Transformer()
    model._rng = None
    model._initial_video_noise = None
    generator = torch.Generator().manual_seed(model.config.seed)
    expected = torch.randn(2, 7, 2, 2, generator=generator)
    chunks = [
        model._video_initial_noise(torch.empty(frames, 2, 2, 2), index)
        for index, frames in enumerate((3, 3, 1))
    ]
    assert torch.equal(torch.cat(chunks).movedim(-3, -4), expected)
    assert model._initial_video_noise.device.type == "cpu"
    assert torch.equal(model.rng.get_state(), generator.get_state())
    renoise = model._rank0_noise(torch.empty(3, 2, 2, 2))
    expected_renoise = torch.randn(2, 3, 2, 2, generator=generator).movedim(-3, -4)
    assert torch.equal(renoise, expected_renoise)


def test_short_final_chunk_keeps_nominal_query_slot_and_history_budget():
    cache = BlockKVCache(
        k_shape=(1, 18, 1, 1),
        v_shape=(1, 18, 1, 1),
        seq_dim=1,
        chunk_size=3,
        window_size=12,
        sink_size=6,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    for index in range(6):
        values = torch.arange(index * 3, index * 3 + 3).float().reshape(1, 3, 1, 1)
        cache.before_update(index)
        cache.update(values, values)
        cache.after_update(index)
    cache.valid_chunk_size = 1
    cache.before_update(6)
    value = torch.full((1, 1, 1, 1), 18.0)
    cache.update(value, value)
    assert cache.write_end == 16
    assert cache.cached_k().flatten().tolist() == list(range(6)) + list(range(9, 19))
    cache.after_update(6)
    assert cache.size == 16


def test_scalar_timestep_time_mlp_uses_batched_gemm():
    network = WanDiTNetworkConfig(
        dim=8, ffn_dim=16, freq_dim=8, num_heads=1, num_layers=0,
        in_dim=2, out_dim=2, text_dim=8, text_len=2, patch_size=(1, 1, 1),
        patch_embedding_type="linear", fp32_adaln=True,
    ).setup()
    network.update_parameters_after_loading_checkpoint()
    shapes = []
    network.time_embedding[0].register_forward_pre_hook(
        lambda module, inputs: shapes.append(tuple(inputs[0].shape))
    )
    network(x=torch.zeros(2, 2), timesteps=torch.tensor(0.0),
            cache=WanDiTNetworkCache(block_caches=[]),
            rope_freqs=torch.empty(2, 1, 1, 8), eager_mode=False)
    assert shapes == [(1, 8)]
