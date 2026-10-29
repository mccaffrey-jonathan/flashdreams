<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Vast AI
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Game2V: EdenLive games rendered into video

Game2V runs an EdenLive three.js game in headless Chromium and feeds its frames
(the *proxy*) to the EdenLive proxy-DINO model, which renders them as detailed
video from a first frame and a prompt. It can also replay a recorded proxy
video instead of a live game, reading it exactly as LightX2V's EdenLive
`RecordingReader` does, so FlashDreams and the reference consume the same pixels.

Slugs: `game2v-eden-world-v2-5b-taehv` (Wan 2.2 TI2V-5B, 1280x704 at 24 fps; fits a
32 GB card), `game2v-eden-world-v2-5b-taehv-rtx-5090` (the same with FP8 linears, FP8
Sage attention, `torch.compile` and CUDA graphs; the first run compiles for several
minutes while the KV window fills) and `game2v-eden-world-v2-14b-taehv` (Wan 2.1
I2V-14B, 832x480 at 16 fps).

## Setup

The games are not part of FlashDreams. Get them from EdenLive and point
`EDEN_GAMES_ROOT` at them (a replay does not need them):

```bash
git clone https://github.com/VAST-AI-Research/EdenLive
npm ci --prefix EdenLive/realtime/games          # three.js for the games
export EDEN_GAMES_ROOT=$PWD/EdenLive/realtime/games
uv pip install "playwright>=1.50"                 # or: uv sync --package flashdreams-eden --extra game --inexact
python -m playwright install chromium
export EDEN_MODEL_ROOT=/path/to/EdenLive          # VAST-AI/EdenLive weights
```

`--game` takes any slug in `$EDEN_GAMES_ROOT/registry.json` (default: the first).
Each game's `cover.png` is the first frame and its registry `prompt` the static
prompt; `--first-frame`/`--image-path` and `--prompt` override them.

## Running

Headless recording with the games' rule-based autopilot (it reads
`<game>/demo.json`), the proxy beside the generated video, the proxy itself and a
per-chunk log:

```bash
flashdreams-run-v2 game2v-eden-world-v2-5b-taehv --mode mp4 \
  --output-path runs/ember.mp4 --total-model-steps 40 -- \
  --game ember-den --total-blocks 40 --autopilot \
  --side-by-side --record-proxy runs/ember.proxy.mp4 --chunk-log runs/ember.jsonl
```

Play it yourself in a local window (keys go to the game; see each game's
`controls` in the registry), or in a browser with the game render, HUD and game
panel beside the generated video:

```bash
flashdreams-run-v2 game2v-eden-world-v2-5b-taehv --mode native-window -- --game jade-duel
flashdreams-run-v2 game2v-eden-world-v2-5b-taehv --mode webrtc -- --game jade-duel   # open the printed URL
```

Replay a recorded proxy (the ComfyUI / LightX2V comparison path):

```bash
flashdreams-run-v2 game2v-eden-world-v2-5b-taehv --mode mp4 \
  --output-path runs/cryo.mp4 --timeout unbound -- \
  --proxy-video samples/cryo-crossing/recording.mp4 \
  --first-frame samples/cryo-crossing/cover.png \
  --prompt "$(python -c "import json; print(json.load(open('samples/cryo-crossing/prompts.json'))['staticPrompt'])")" \
  --seed 2026092901 \
  --max-frames 117 --record-proxy runs/cryo.proxy.mp4 --side-by-side
```

On Windows add `--no-compile` unless the Triton/Inductor warmup is wanted.

### App flags (after `--`)

| Flag | Meaning |
| --- | --- |
| `--game SLUG` | Game to render (registry slug). |
| `--autopilot [SCRIPT]` | Play headless; default `$EDEN_GAMES_ROOT/_shared/autopilot.js`. `EDEN_GAME_AUTOPILOT` still works. |
| `--proxy-video PATH` | Replay this video instead of a game. Needs `--first-frame` and `--prompt` (or `--prompts-json`). |
| `--max-frames N` | Cap a replay at `N` frames before rounding to `4n+1`. |
| `--prompts-json PATH` | EdenLive `prompts.json`: chunk `c`'s prompt is `staticPrompt` plus the `dynamicPrompts` overlapping its frames (EdenLive's offline `prompt_schedule`). Keeps the text encoder resident when the text changes. |
| `--record-proxy PATH` | MP4 of every proxy frame the model consumed, aligned 1:1 with the output frames. |
| `--side-by-side` | Present `proxy \| generated` (twice the model width). |
| `--chunk-log PATH` | One JSON line per chunk: frames, condition frames, prompt in effect, event lines, timings, last HUD state. |
| `--total-blocks N` | AR chunks per rollout (default 20; a replay defaults to the whole video). |
| `--no-compile` | No `torch.compile` / CUDA graphs anywhere (DiT, VAE, decoder, DINO). |
| `--dynamic-prompt` | Re-encode each chunk's prompt with the live game's event lines. |

### Replay rules

A replay generates `N = 4*((floor(duration*fps) - 1)//4) + 1` frames (after
`--max-frames`), i.e. `T = (N-1)/4 + 1` latent frames in `T/3` chunks; `T` must be
a multiple of 3 (the error names the `--max-frames` that is). The model is
conditioned on **global** frames `0, 4, 8, ...` (one per latent frame), decoded
with PyAV as `rgb24`, nearest-midpoint frame selection, centre crop and PIL
bilinear only when the video is not the model size. The whole video's initial
noise is drawn in one piece (`max_latent_frames = T`), which reproduces the
reference RNG stream for a seed. A live game instead conditions on frames
`0, 4, 8` *of each chunk*, so do not compare a live run's proxy recording frame
for frame with a replay of it.

## How the live proxy works

* One Chromium page per session (rank 0), on a virtual 60 Hz clock
  (`$EDEN_GAMES_ROOT/host/standalone.js`), captured at the model's frame size as
  JPEG. Chunk `N+1` is rendered while chunk `N` denoises, stretched over the
  last generate so keys land in the in-flight simulation; its DINOv3 encode is
  chained onto the render. Press-to-visible is about two chunks.
* Keys: the UI loop forwards every keyboard edge it reads to a queue the capture
  worker drains between frames (the model loop itself only sees input at step
  boundaries). This needs the default `--presentation-mode continuous`; with
  `on_demand` the UI only steps when a frame is presented.
* The model loop returns two channels: the generated video and the proxy render
  of the same frames. The UI loop presents channel 0 (or both, side by side).
* Rasterization (`EDEN_GAME_RASTER`): on Windows the default is `warp`
  (`--use-angle=d3d11-warp`, Windows' multi-threaded CPU rasterizer). With the game
  on the same GPU as the model (`gpu`, `--use-angle=d3d11`), the canvas readback
  waits for GPU time slices the DiT holds (up to ~2 s a frame), which capped the
  rtx-5090 preset's live loop at ~11 fps; WARP runs it at ~17 fps with frames
  within ~52 dB of the GPU render. Linux defaults to `gpu` (NVIDIA's EGL library
  registered with GLVND, `--use-angle=gl-egl`). `swiftshader` (or
  `EDEN_GAME_GPU=0`) is Chromium's slower CPU fallback, also used when the page
  gets no WebGL. The chosen renderer is logged, e.g.
  `Game2V capture WebGL renderer: ANGLE (Microsoft, Microsoft Basic Render Driver ... D3D11)`.
* Pacing: in interactive modes the next chunk's render is stretched over the last
  generate so live keys land in it; `--mode mp4` has no live input and renders
  ahead as fast as it can (WARP plus no pacing: ~19 fps on the rtx-5090 preset,
  the model's own rate).
* `--capture-size WxH` renders the game smaller and upscales it to the model size.
  It cuts browser work but not the GPU-contention stalls (832x464 on the GPU:
  ~11 fps, like full size).
* mp4 mode presents on demand, so a win/lose card never pauses the rollout; after
  the round ends the game holds its end state. Size `--total-blocks` to the play.

## Browser window (`eden-webrtc`)

With the `eden-webrtc` client-window mode installed (entry point in the
integration's `pyproject.toml`), `--mode webrtc` serves FlashDreams' own page plus
`/eden.js`: the generated video with the game's HTML HUD overlay
(`$EDEN_GAMES_ROOT/<game>/hud/overlay.js`), the scene prompt and the chunk's event
lines, the game render, and the `/ext/game2v` game panel (switch, preview, import).
The UI sends the side-by-side canvas as the one video track and the page crops it
into the two panels, so the game render and the generated frame are paired exactly.
HUD buttons (`data-game-action`, e.g. *Play again*) reach the game as a key pulse.
It binds `--host 127.0.0.1` by default; its extra routes are unauthenticated, as in
EdenLive, so do not expose it beyond a trusted network. Not ported: TURN, the
heartbeat that pauses generation without a client, and the frame replay on client
takeover (a reconnecting client is re-sent the current prompt and HUD state only).

## Importing a generated game

`python -m eden.apps.game2v.import_game /path/to/project --slug star-course
--games-root $EDEN_GAMES_ROOT [--cover photoreal.png]` registers an EdenGolem
(GamePrototypeTurbo) standalone project; without `--cover` it captures the game's
first frame (a greybox anchor). The EdenGolem skill itself is part of EdenLive.

## Tests

```bash
pytest integrations_v2/eden/tests -m ci_cpu                      # no games, weights or GPU
EDEN_GAMES_ROOT=... pytest integrations_v2/eden/tests -m ci_cpu  # + registry tests
EDEN_GAMES_ROOT=... pytest integrations_v2/eden/tests/test_game2v_chromium.py -m manual
```

Set `EDEN_LIGHTX2V_RECORDING_PY` to LightX2V's
`lightx2v/models/input_encoders/edenlive/recording.py` to also compare the replay
reader against it directly.
