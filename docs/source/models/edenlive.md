---
title: 'EdenLive'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

<div class="fd-cta-row">
  <a class="fd-button" href="https://vast-eden.ai/?p=edenlive">Project page</a>
  <a class="fd-button" href="https://github.com/VAST-AI-Research/EdenLive">Official code</a>
</div>

Introduced by [VAST](https://www.tripo3d.ai/) (Project Eden), EdenLive is a real-time generative renderer: it repaints a
*reference* video -- a greybox game, a capture, any rough render -- into detailed frames, guided by a target first
frame and a text prompt. The upstream program executes the rules and owns the world state; EdenLive renders it. The
reference reaches a Wan 2.2 TI2V-5B (or Wan 2.1 I2V-14B) DiT through a post-trained DINOv3 encoder.

<div class="fd-card fd-model-card">
  <img class="fd-card-preview" src="../_static/model_clips/edenlive/edenlive-cryo-crossing.avif" alt="" />
</div>
<figcaption class="tiny-figcaption">
  Left: greybox proxy (EdenLive <code>cryo-crossing</code> sample). Right: <code>game2v-eden-world-v2-5b-taehv-rtx-5090</code>
  on one RTX 5090, recorded-proxy replay, seed 2026092901.
</figcaption>

## Quick Start

### Game2V

```bash
uv sync --package flashdreams-eden --inexact
hf download VAST-AI/EdenLive --local-dir models/EdenLive \
  --include "dit-5b/*" --include "vae-5b/*" --include "decoder-5b/*" \
  --include "scheduler-5b/*" --include "umt5/*" --include "vision-encoder/*"
export EDEN_MODEL_ROOT=$PWD/models/EdenLive HF_HUB_OFFLINE=1

# Proxy games live in the EdenLive repository.
git clone https://github.com/VAST-AI-Research/EdenLive && npm ci --prefix EdenLive/realtime/games
export EDEN_GAMES_ROOT=$PWD/EdenLive/realtime/games
uv sync --package flashdreams-eden --extra game --inexact
uv run --no-sync python -m playwright install chromium

uv run --no-sync flashdreams-run-v2 game2v-eden-world-v2-5b-taehv-rtx-5090 --mode native-window
```

<hr>

## Demo Presets

<a id="game2v-presets"></a>

### Game2V

| Preset | Description |
| --- | --- |
| `game2v-eden-world-v2-5b-taehv` | EdenLive 5B, bf16, LightTAE decoder. 704x1280 at 24 fps. |
| `game2v-eden-world-v2-5b-taehv-rtx-5090` | The 5B with row-wise FP8 linears, in-tree Sage-style FP8 attention, torch.compile and CUDA graphs, for 32 GB GeForce cards. |
| `game2v-eden-world-v2-14b-taehv` | EdenLive 14B (Wan 2.1 I2V), 480x832 at 16 fps. Needs more than 32 GB. |

Measured on one RTX 5090 (Windows 11, PyTorch 2.12.1+cu130), recorded-proxy replay of the 717-frame
`cryo-crossing` sample, steady state once the 18-frame attention window is full:

| Preset | Throughput | Peak VRAM |
| --- | --- | --- |
| `game2v-eden-world-v2-5b-taehv` (eager) | 10.3 fps (1.17 s per 12-frame chunk) | 22.6 GiB |
| `game2v-eden-world-v2-5b-taehv-rtx-5090` | 18.6 fps (0.64 s per 12-frame chunk) | 18.1 GiB |

The FP8 preset compiles one graph per KV-fill shape on its first run (a few minutes); later runs reuse the
Inductor cache.

<hr>

## Developer Details

<div class="fd-cta-row">
  <a class="fd-button" href="https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/eden">Integration Source</a>
  <a class="fd-button" href="https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/eden/config.py">Pipeline configuration</a>
</div>

- **Minimum VRAM:** 24 GB for the bf16 5B preset, 20 GB for the RTX 5090 preset.
- **Disk:** ~37 GB for the 5B weights (skip the `flux-c2f/` tree, which is unused and non-commercial).
- **Licenses:** code Apache-2.0. The DiT, VAE, UMT5 and LightTAE weights are Apache-2.0; the required
  `vision-encoder/` is under the [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md),
  which allows commercial use but is not OSI-approved (trade-control and prohibited-use terms). See
  [THIRD-PARTY-NOTICES](https://github.com/NVIDIA/flashdreams/blob/main/THIRD-PARTY-NOTICES).
