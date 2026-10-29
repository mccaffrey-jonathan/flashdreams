<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Vast AI
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# EdenLive (Game2V)

[EdenLive](https://github.com/VAST-AI-Research/EdenLive) (VAST, Project Eden) is a
real-time generative renderer: it repaints a *reference* video -- a greybox
game, a capture, any rough render -- into detailed frames, guided by a target
first frame and a text prompt. The reference reaches the DiT through a
post-trained DINOv3 encoder whose 1024 projected channels are concatenated onto
the latent channels. The upstream program owns the game state; EdenLive only
renders it.

This package ports VAST's FlashDreams-based real-time integration
(`realtime/integrations_v2/eden`, forked from FlashDreams at `c1889e04`) onto
FlashDreams v0.2.0, restricted to the Game2V path.

## Pipeline configurations (`eden.config`)

| Config | Model | Resolution | Decoder |
| --- | --- | --- | --- |
| `PIPELINE_GAME2V_EDEN_WORLD_V2_5B` | Wan 2.2 TI2V-5B, chunk 3, 4 steps | 704x1280 @ 24 fps | Wan 2.2 VAE |
| `PIPELINE_GAME2V_EDEN_WORLD_V2_5B_TAEHV` | same | same | LightTAE `lighttaew2_2` |
| `PIPELINE_GAME2V_EDEN_WORLD_V2_14B` | Wan 2.1 I2V-14B, chunk 3, 4 steps | 480x832 @ 16 fps | Wan 2.1 VAE |
| `PIPELINE_GAME2V_EDEN_WORLD_V2_14B_TAEHV` | same | same | LightTAE `lighttaew2_1` |

The numbers live in `configs/game2v_{5b,14b}.yaml`. Both models use 4
self-forcing steps (5B timesteps `[1000, 937.5, 833.33, 625]`), CFG off, an
18-frame attention arena (6 sink + 12 rolling) with cache-slot RoPE, and the
first frame stamped into chunk 0 at `t=0`.

The 14B does not fit a 32 GB card at the published window (about 30.5 GiB of
bf16 weights plus 21.4 GiB of KV cache); the 5B does.

### Precision and kernel switches

The network configs expose three switches, all off by default except where the
YAML turns them on:

* `fp32_adaln` (YAML: on) -- Diffusers-precision AdaLN / LayerNorm / residuals,
  the numerics the checkpoints were evaluated with (`impl/numerics.py`).
* `linear_backend="rowwise_fp8"` -- dynamic row-wise FP8 E4M3 block linears.
* `self_attention_backend="sage_fp8"` (in-tree Sage-style INT8-QK / FP8-PV
  Triton kernel) or `"sage2"` (external SageAttention 2 package). Single GPU.

### Differences from VAST's fork

VAST patched FlashDreams core in place; here every model-specific change is an
`eden.impl` subclass, so other integrations keep their numerics:
`EdenFlowMatchScheduler` (fp32 network timesteps, bf16 sigma),
`EdenDiffusionModel` (reference `(C, T, H, W)` noise layout, bounded whole-video
noise), `TaehvVAEWan22Decoder` (48-channel LightTAE), `EdenWanVAEEncoder`
(fp32 RMS norm; also the full-VAE decoders), `EdenCLIPImageEncoder` (14B: uint8
cover preprocessing), scalar-timestep KV commit. Not ported: the fork's switch of the
Wan attention backend to `flash` (PyTorch's Windows wheels ship without
FlashAttention; the cuDNN backend stays), its fp64 RoPE kernel, fp32 sinusoidal
time embedding and real-`Conv3d` patch embedding (small numerics-only changes
that were unconditional for every Wan model), the multi-GPU orchestration
(v0.2.0 has its own), and the core -> integration `EDEN_MODEL_ROOT` import.

## Licenses

* **Code**: Apache-2.0 (VAST EdenLive, Copyright (c) 2026 Vast AI; FlashDreams,
  Copyright (c) 2026 NVIDIA).
* **Weights** (`VAST-AI/EdenLive` on Hugging Face, downloaded separately):
  * `dit-*`, `vae-*`, `umt5/`, `scheduler-*`, `decoder-*`: Apache-2.0.
  * `clip/` (14B only): MIT (LAION CLIP ViT-H/14).
  * `vision-encoder/` (required by every EdenLive model): the
    [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
    -- royalty-free and commercial use allowed, but **not** an OSI license:
    trade-control and prohibited-use terms (no military, ITAR, weapons, nuclear
    or espionage use), no reverse engineering, pass-through on redistribution,
    and research publications must acknowledge DINOv3.
  * `flux-c2f/`: FLUX.2 [klein] Non-Commercial License. **Not used by this
    integration** -- it only powered VAST's workbench first-frame renderer,
    which is not ported. Skip it when downloading.
* **Not ported**: VAST's workbench (FLUX first-frame worker, and a game-authoring
  bridge that drives the `claude` CLI with `--dangerously-skip-permissions`) and
  the `cam2v-eden-*` presets (a renamed copy of `integrations_v2/lingbot` that
  loads LingBot-World v2 weights, CC BY-NC-SA 4.0).
* **Games**: the three.js proxy games stay in the EdenLive repository
  (`realtime/games`, Apache-2.0 with MIT-licensed GameBlocks modules and the
  html-to-image bundle); point `EDEN_GAMES_ROOT` at a checkout.

## Install and weights

```bash
uv sync --package flashdreams-eden --inexact
# Game2V live mode (headless Chromium for the proxy games):
uv sync --package flashdreams-eden --extra game --inexact
uv run --no-sync python -m playwright install chromium

# 5B weights only (~37 GB), without the non-commercial FLUX tree:
hf download VAST-AI/EdenLive --local-dir models/EdenLive \
  --include "dit-5b/*" --include "vae-5b/*" --include "decoder-5b/*" \
  --include "scheduler-5b/*" --include "umt5/*" --include "vision-encoder/*"
export EDEN_MODEL_ROOT=$PWD/models/EdenLive HF_HUB_OFFLINE=1
```

`EDEN_MODEL_ROOT` (or `model_root` in the YAML) names the weight tree.

## Game2V application

```bash
git clone https://github.com/VAST-AI-Research/EdenLive && npm ci --prefix EdenLive/realtime/games
export EDEN_GAMES_ROOT=$PWD/EdenLive/realtime/games
uv run --no-sync flashdreams-run-v2 game2v-eden-world-v2-5b-taehv --mode native-window -- --game <slug>
```

See [`apps/game2v/README.md`](apps/game2v/README.md) for controls, headless
recording (`--mode mp4`, autopilot) and the recorded-proxy replay mode.

## Programmatic pipeline access

```python
from eden.config import PIPELINE_GAME2V_EDEN_WORLD_V2_5B_TAEHV

pipeline = PIPELINE_GAME2V_EDEN_WORLD_V2_5B_TAEHV.setup().to("cuda").eval()
cache = pipeline.initialize_cache(text=[prompt], image=cover)  # cover: [1,3,704,1280] in [-1,1]
for ar in range(blocks):
    control = dino.encode(proxy_frames[3 * ar : 3 * ar + 3], latent_height=44, latent_width=80,
                          dtype=torch.bfloat16)               # eden.impl.game14b.dino
    video = pipeline.generate(autoregressive_index=ar, cache=cache, input=control)
    pipeline.finalize(ar, cache)
```

## Tests

```bash
uv run --no-sync pytest integrations_v2/eden/tests -m ci_cpu
```
