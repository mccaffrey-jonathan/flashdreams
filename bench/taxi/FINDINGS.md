# Crazy Robotaxi on a 32 GB GeForce RTX 5090

Headless runs: `--mode mp4 --no-ui`, 40 blocks, `boulevard_district`, profiling
on; WSL2 unless noted. Per-run stats are under `runs/<tag>/` (not committed);
`LEADERBOARD.md` is the table.

## Results

No stock preset fits. Two new presets do (8-frame chunks; real time needs
a model-thread step of 267 ms or less):

| runner | frame | model step | model-thread wall | end to end | process peak |
|---|---|---|---|---|---|
| `rtx-5090` (WSL2) | 1168×640 | 258 ms | 375 ms | 21 fps | 27–31 GiB at prewarm, 21.7 GiB steady |
| `rtx-5090-fast` (WSL2) | 1024×560 | 156 ms | 262 ms | 30.5 fps | 25.3 GiB |
| `rtx-5090-fast` (Windows, `--no-ui`) | 1024×560 | 194 ms | 257 ms | 31.1 fps | 15.3 GiB torch reserved |
| `rtx-5090-fast` (Windows, native window + HUD) | 1024×560 | 204 ms | 277 ms | 28.7 fps | 15.2 GiB torch reserved |

Windows uses cuDNN FP8 attention (SageAttention-3 is Linux-only). The stock
`perf` preset on this GPU runs out of memory during prewarm; with only the
text-encoder fix it peaks at 32 GB and WSL's unified memory spills to host RAM,
leaving it at 2 to 6 fps.

## Findings

1. **HUD needs Vulkan even for `--mode mp4`.** The ImGui loop creates a SlangPy
   Vulkan device; WSL2 has no NVIDIA ICD. `--no-ui` presents the raw video
   channel and keeps the HUD state, so `--game-mode`/`--map` startup still runs.
2. **The Cosmos-Reason1-7B text encoder is 15.45 GiB of the 19.4 GiB load
   footprint** (DiT: 3.84 GiB). Game hosts keep the pipeline resident and
   re-encode the scene prompt on every restart, so they cannot release it, and
   swapping it back onto the GPU needs 15 GiB free. `run_on_cpu` encodes a
   512-token prompt in 2.2 s on a 20-core desktop CPU with zero VRAM;
   `embedding_cache_size` makes restarts free.
3. **A ~7.5 GiB allocation outside the torch allocator, every other AR step
   once the KV window is full,** takes the process to exactly 32 GB and WSL
   spills; once spilled, diffuse is 1.17 s and the conditioning renderer 2.1 s
   per chunk, permanently. It scales with `window_size_t` and is independent
   of the attention backend and the cuDNN plan policy; `torch.cuda.empty_cache()`
   does not help. `window_size_t=4` (what the `-responsive` presets use) is the
   fix. Measured with `mem_probe.py` (driver-used minus torch-reserved, tagged
   by pipeline phase); the same probe puts 1280×704 with window 4 at 23.9 GiB
   without the game engine.
4. `sage3_fp8` attention: diffuse 119 ms vs 157 ms for cuDNN at 1024×560. The
   native FP8 LightVAE (`fast-perf` base) cuts encode from 52 ms to 17 ms; its
   calibration auto-exports on first run.
5. **The game engine costs 60 to 100 ms per chunk on the model thread**
   (`physx_traffic_prepare` 48 ms + conditioning render 30 ms on WSL2; 39 + 21 ms
   on Windows). It is what keeps 1168×640 at 21 fps; overlapping engine and
   model work is the next step.

## Notes

- `--stats-path` recorded only the pipeline finalize metrics for this app; the
  model loop now attaches the full per-step dict (engine, PhysX, realtime margin).
- WSL2: `ludus_renderer_plugin` links `-lcuda`, which lives in `/usr/lib/wsl/lib`;
  `taxi_bench.sh` exports `LIBRARY_PATH` accordingly.
- The app's `--width/--height` must match run-v2's `--pixel-width/--pixel-height`.
- Windows: a killed run leaves torch's `FileBaton` `lock` in
  `integrations_v2/omnidreams/impl/omnidreams_singleview/build/torch_extensions/<ext>/`;
  the next launch then waits forever right after the first LightVAE checkpoint
  log line (idle CPU, no `--timeout`). Delete the lock.
- Windows: keep `HF_HOME` short. The `omni-dreams-samples` file paths exceed
  MAX_PATH under a long cache root, which surfaces as a silently missing file;
  junctions do not help because huggingface_hub resolves them.

## Harness

`taxi_bench.sh TAG SLUG [run-v2 args] -- [app args]`, `parse_stats.py`
(→ `ledger.jsonl`), `leaderboard.py` (→ `LEADERBOARD.md`), `mem_probe.py`,
settings overlays in `cfg/`, `win_run.cmd` (MSVC environment for the native
build on Windows), `win_drive_capture.ps1` (screen-captures the native window
while scripted key presses drive).
