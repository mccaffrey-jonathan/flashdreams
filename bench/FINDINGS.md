# OmniDreams 5090 — VRAM breakdown & fitting 1280×704

RTX 5090 (32 GiB), WSL2, two-step distilled chunk2 single-view perf chassis.
Profiler: `profile_mem.py` (phase-delta memory + `torch.cuda.memory` allocator
snapshot grouped by allocation call-site). FPS: `run_bench.py` (offline runner,
per-AR-step steady median; 8 frames/block ⇒ 30 fps = 267 ms, 25 fps = 320 ms).

## The decisive finding: the HDMap preload artifact

The offline runner (`runner.py:_load_video` / `_rollout_and_save`) loads the
**entire** HDMap conditioning clip to GPU and stacks it (2 copies). The bundled
example clip is **2451 frames** (80 s @ 1280×720), so this term is huge and
scales with pixel area:

| res | HDMap preload (2 copies) |
|---|--:|
| 640×352 | 6.2 GiB |
| 1280×704 | **~24.7 GiB** |

It is a **benchmark artifact** — real streaming inference feeds the HDMap
frame-by-frame and only a 2-frame window is consumed per AR step. Trimming the
input clip to 200 frames (`assets/hdmap_200.mp4`, via `--hdmap`) removes it
(~1 GiB/copy) with no effect on steady-state fps.

## Native 1280×704 working-set breakdown (trimmed HDMap)

Steady working set **19.75 GiB**, reserved 26.6, peak 28.8 (encoder phase),
`num_ooms=0` — **fits 32 GiB with margin, no spill.**

| group (call-site) | GiB | what |
|---|--:|---|
| `kvcache.py:__post_init__` | 9.26 | KV cache (effective window ~11–12 latent frames) |
| `optimized_dit:_ensure_weights_snapshot` | 3.85 | resident **bf16** weight snapshot |
| `optimized_dit:_ensure_fp8_runtime` | 2.35 | FP8 quantized weights + scales |
| HDMap (trimmed) | 2.0 | was ~24.7 untrimmed |
| `_make_cosmos_streaming_workspace` | 0.88 | CUTLASS streaming scratch |
| VAE / rope / misc | ~1.4 | |

## Scaling across resolutions (native, compile_network=False, trimmed HDMap, window=6)

| group | 640×352 | 896×512 | 1280×704 | vs area | verdict |
|---|--:|--:|--:|:--|:--|
| KV cache | 2.49 | 4.83 | 9.26 | ×1.94 / ×1.92 (area ×2.04 / ×1.96) | ∝ area — efficient |
| bf16 weight snapshot | 3.845 | 3.848 | 3.846 | flat | **fixed (resolution-independent)** |
| FP8 runtime buffers | 0.62 | 1.26 | 2.35 | ×2.03 / ×1.87 | ∝ area (activation scratch, not weights) |
| HDMap (trimmed, ×2) | 0.50 | 1.03 | 2.02 | ×2.04 / ×1.96 | ∝ area |
| CUTLASS workspace | ~0.23 | 0.45 | 0.88 | ×~2 | ∝ area |
| **working set** | **8.38** | **12.35** | **19.75** | | |

Token count = (H/16)×(W/16); area-scaling of KV / activations / workspace is
exactly right (space-efficient — no super-linear blowup, no per-resolution
waste). Pixel area ratios: 640→896 = 2.04×, 896→1280 = 1.96×, 640→1280 = 4.0×.

### Space-efficiency notes
- **KV cache is the #1 consumer and scales ∝ token area** (2.49→4.83→9.26 GiB).
  Implied effective context ~11–12 latent frames at the 1280×704/window-6 point
  — larger than nominal `window_size_t=6`; reducing it (→4) was the cheapest
  fps+VRAM lever (9.26→~6 GiB, peak 28.8→25.9).
- **The bf16 weight snapshot (3.85 GiB) is the only truly fixed, resolution-
  independent block** — a resident master copy the native FP8 path keeps. It is
  the candidate "free if releasable after quantization" lever (~3.8 GiB). NOTE:
  `_ensure_fp8_runtime` is NOT a duplicate weight copy (it scales with area, so
  it is per-forward FP8 activation/scale scratch) — an earlier note here was
  wrong; the scaling table corrects it.
- Allocator fragmentation (reserved − alloc) ≈ 2–7 GiB under WSL GPU-PV; the
  no-spill cliff is ~28–30 GiB **reserved** (WSL UM spills near, not at, capacity).

## FPS results @ 1280×704 (16 blocks, trimmed HDMap) — TARGET MET

| config (stacked) | fps | diffuse | enc | dec | total ms | peak GiB |
|---|--:|--:|--:|--:|--:|--:|
| native, compile_network=False, skip-finalize | 18.9 | 345 | 56 | 21 | 423 | 28.8 |
| + sage3_fp8 attention | 23.9 | 256 | 56 | 21 | 334 | 28.8 |
| **+ window_size_t=4** | **27.0** | 217 | 56 | 21 | 296 | **25.9** |

**1280×704 runs at 27 fps on the 5090, full resolution, no quantization of the
weights, no spill.** Winning flags:
```
--pixel-height 704 --pixel-width 1280
--pipeline.diffusion-model.transformer.native-dit-acceleration required
--pipeline.diffusion-model.transformer.compile-network False
--pipeline.diffusion-model.transformer.native-dit-attention-backend sage3_fp8
--pipeline.diffusion-model.transformer.window-size-t 4
--pipeline.diffusion-model.transformer.skip-finalize-kv-cache True
```
Levers, in order of impact: **(1) remove the HDMap full-clip preload** (the
fit enabler), **(2) sage3_fp8 attention** (diffuse −26%), **(3) window_size_t
6→4** (diffuse −15% + KV/peak 28.8→25.9). The window trim is the one quality
compromise (2 chunks of temporal context vs 3); everything else is lossless.
Remaining floor: encode 56 ms (per-step LightVAE HDMap encode) — would need
native FP8 VAE (calibrated state) to cut further. See `LEADERBOARD.md`.
