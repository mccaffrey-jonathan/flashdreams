# Eden Game2V backend games

How the proxy render behind the model works: the Chromium game, its capture
path, the HUD channel and the timing. For how to start a session, see
[the deployment README](../../../../README.md).

The registered games live in `games/registry.json` at the checkout root. Run
installation commands in the same Python environment used by `torchrun`:

```bash
pip install -e ./integrations_v2/eden
pip install playwright
python -m playwright install chromium
npm install --prefix games
```

Use `EDEN_GAMES_ROOT` for a different resource directory. `EDEN_THREE_MODULE`
can override the workbench Three.js 0.161.0 build. The seven bundled games use
`three-demo` (an npm alias for Three.js 0.186.1), installed by the same command. Games are checkout resources, not included in the Eden wheel.
Playwright is loaded lazily only by Eden's rank-zero model loop.

## Bundled games

The seven games ship with this checkout and run locally: Cryo Crossing (`freeze`),
Spider Hollow (`spiders`), Pulse Skirmish (`tracer`), Deepwater Shock
(`deepwater-shock`), Kaiju Rampage (`kaiju-rampage`), Arcade Duel (`arcadeduel`),
and Desert Crossfire (`desert-crossfire`). They are listed in
`games/registry.json`. No separate demo server is required.

Each game keeps its gameplay and scene modules, `cover.png`, the matching
`first-frame.png` proxy, and `initial-state.json`. `host/demo-runtime.js` adapts
these modules to `createRuntime()`: it accumulates the host's 60 Hz time and steps
the game at its native 24 Hz. Captures remain at the selected model FPS. The
adapter collects every intervening state for `dynamicPrompt()` and publishes the
result through the existing HUD/event channel. Enable `--dynamic-prompt` to feed
these descriptions back into the model.

Start and replay restore the bundled initial state. The **Play again** HUD button
requests a full rollout restart, including the model cache and reference frame.
R remains available for reload or special abilities. P/Escape pauses and resumes.
The proxy canvas contains no HUD; health, resources, objectives and outcomes are
separate overlays over the displayed video. The live proxy preview uses the same
adapter and controls.

The games use Three.js 0.186.1 through `/vendor/three-demo.module.js`; generated
workbench projects retain their existing Three.js 0.161.0 contract. Both versions
are pinned in `games/package.json`.

Validation (inside the runtime pod):

```bash
for test in games/tests/*.mjs; do node "$test" || exit; done
python integrations_v2/eden/tests/check_game_backend.py --render
python integrations_v2/eden/tests/check_game_preview.py
pytest integrations_v2/eden/tests -m ci_cpu
```

## Importing a generated game

`skills/edengolem/` is a vendored copy of the GamePrototypeTurbo
skill. Given a game description, an agent following its `SKILL.md` builds a
standalone Three.js project with reachable win and loss states. Generating the
project takes an agent. Registering the finished project is one command:

```bash
python -m eden.apps.game2v.import_game /path/to/project --slug star-course \
  --cover /path/to/photoreal-cover.png        # optional, see below
```

The workbench home screen does this end to end -- describe a game, and it
generates, renders a first frame and registers the result. The command above is
the same step for a project built elsewhere.

The importer copies the run that passed acceptance (`--allow-incomplete` takes
the latest run instead), along with its `assets/` closure, into
`games/<slug>/`. It also writes a minimal HUD overlay with a result card and a
restart button, and appends a registry entry with `host: "standalone"`. The
brief supplies the title, the static prompt (`scene.description`, override with
`--prompt`) and the description. The run card supplies the controls, and the
strategy supplies the restart key. Hand-ported games cannot be replaced by an
import. Two failure cases are handled: if any step fails, the registry and any
replaced directory are restored, and an existing slug requires `--replace`.

Skill games own their loop, so they are not rewritten into `createRuntime()`.
Instead `GameRenderer` injects `games/host/standalone.js` ahead of the page's
modules. That script replaces `requestAnimationFrame` and `performance.now()`
with a virtual clock. Each simulation tick is exactly one RAF callback 1/60 s
after the previous one, and nothing runs between ticks. The script also
publishes the same `window.gameBackend` used by the hand-ported games. Game
state comes from the game's `window.__gameTest.read()`, which the skill's
acceptance gate already requires. Keys reach the page as `KeyboardEvent`s with
both `key` and `code` set, and a `restart` client action presses the game's own
restart key. Event prompt lines come from `--event-prompts`, a JSON map of
event type to sentence. Events without an entry add nothing to the prompt.
Gameplay timed with `setTimeout` is outside the skill's clock contract and
will drift.

Without `--cover`, the importer renders the game's first frame and uses it as
`cover.png`. That makes the rollout's anchor frame a greybox image, so pass a
photoreal render of the opening view whenever one exists.

Simulation advances at fixed 1/60-second steps. Captured frames are the
model's own `frames_per_second_for_step` (Eden default 16, overridable with
`--fps`): a 12-frame chunk at 16 fps covers 0.75 s = 45 ticks, at 24 fps
0.5 s = 30 ticks. The page passes that tick delta into `runtime.render`, so
visual interpolation follows the model instead of a hardcoded 24 fps.
Keyboard and HUD clicks arrive at any wall-clock time, mutate the live
Chromium game, and are sampled into the next captured model frame. There is
no 2 fps hold-last sampler and no chunk-start freeze of `input_events`.

A single Chromium page renders every model frame, and that one render serves
both consumers:

* every frame streams to the client's small game window (`ui_frame`), and
* the latent-aligned subset `0, 4, 8, 12, ...` conditions the model
  (`condition_frame`, which references the same render as `ui_frame`).

There is no second page for the sparse condition frames, so the model and the
client see one rasterization rather than two independent ones.

The render is pipelined one chunk ahead of generation and stretched to the
last `generate_step` duration. Each step collects the chunk begun during the
previous step, submits the next chunk (paced so its wall time ≈ last
generate; the first successor uses 0.4 s until a generate has been timed), then runs `generate_step`. The Playwright worker drains a dedicated
EventBuffer reader (`GAME_EVENT_READER_ID = 2`) between frames, so keys that
arrive during generate N land in in-flight chunk N+1. Rank zero owns the
browser; other ranks only advance the tick timeline.

With `T_sim ≈ T_gen ≈ T_play`, press-to-visible-frame is a stable ~2 chunk
delay (the matching streamed frame, including playback). The first step of a
rollout has no predecessor, so it bursts with `duration_s=0`. The last step
does not submit a successor.

The HUD is not pixels: the page also emits a HUD snapshot per frame (from the
game's transport module) that the WebRTC client draws as HTML over the
generated video; the streamed JPEG is the raw scene canvas. Frames are encoded
at the model's own size (832x480 by default, `DEFAULT_CAPTURE_WIDTH`/`HEIGHT`).
Each model frame is its own `advanceToFrame` round-trip so live keys can land
between frames.

`GameCaptureWorker` owns one Chromium on one dedicated thread
(`game2v-render`). Rendering a full 12-frame chunk is ~100–123 ms of work on
GPU, stretched across ~430 ms of generate so the leftover idle tail also eats
keys. One GL context costs about 40 MiB on the rank-zero GPU.

`Game2VModelState.last_game_chunk.frames` retains those CPU RGB uint8 arrays,
frame IDs and simulation timestamps after `generate_step`. The current LingBot
adapter still consumes fixed camera poses; it does not consume game conditions.
Only rank zero renders. A future distributed conditioning adapter must distribute
conditions to its participating ranks explicitly.

Presentation consumes model/game channels together and WebRTC drops queued pairs
as a unit with matching timestamps. The second native browser video receives the
game image. This provides source-frame pairing, not a guarantee of pixel-exact
simultaneous display across two browser video elements. Postprocessing is
currently rejected explicitly to avoid mispairing frames when a processor buffers
or changes cadence.

The Cam2V AR log reports the residual render wait and generation:

```
Cam2V AR 44 [steady] | 12 frames | capture 0.4 ms (eval 110.0, decode 12.0) | generate 430.5 ms | 27.87 fps
```

`generate` and the reported `fps` are model-only. `capture` is the leftover
wait for the chunk begun during the previous step — ~0 in steady state because
the paced render hides behind generate; `eval`/`decode` are that render's own
Playwright and JPEG times even when the wait is ~0. Metrics:
`game_capture_wall_s`, `generate_wall_s`.

Raster and JPEG work runs ahead of generation; pacing sleeps keep the game
responsive to keys between captures. Rendering cost depends on the game,
resolution and available WebGL backend.

`GameRenderer` registers NVIDIA's EGL library for Chromium at launch. GPU
containers ship `libEGL_nvidia.so.0` but register it only when
`NVIDIA_DRIVER_CAPABILITIES` includes `graphics`; without the vendor JSON, GLVND
reports Mesa alone, Mesa has no DRI node, and Chromium silently falls back to
SwiftShader, where rasterization and image read-back are slower. The renderer
writes its own vendor JSON to a temporary directory and passes
`__EGL_VENDOR_LIBRARY_FILENAMES` plus
`--use-angle=gl-egl`, so no system directory is touched and the pod needs no
change. The variable replaces the vendor list rather than extending it, so Mesa
is named explicitly to stay available. A missing or unusable library makes the
loader degrade to software instead of failing, which is why this is applied
unconditionally; set `EDEN_GAME_GPU=0` to skip it. The chosen renderer is logged
once per session, so a silent fallback to software is visible:

```
Game2V capture WebGL renderer: ANGLE (NVIDIA Corporation, NVIDIA L20Z/PCIe/SSE2, OpenGL ES 3.2)
```

Switching rasterizers changes antialiasing and shadow sampling slightly, so
condition frames are not pixel-comparable with earlier software-rendered runs.
The single GL context costs about 40 MiB on the rank-zero GPU and, being a
graphics rather than compute context, does not appear in
`nvidia-smi --query-compute-apps`.

Validation without model loading:

```bash
python integrations_v2/eden/tests/check_game_backend.py
python integrations_v2/eden/tests/check_game_backend.py --render
python integrations_v2/eden/tests/check_game_preview.py
```

`--render` checks all seven games through the real Chromium backend: bundled
initial states, 24/16 FPS capture timing, JPEG output, prompt events, pause and
resume, and native demo victories. The preview check also exercises the HTTP
routes, shared modules, HUD and restart request payload.
