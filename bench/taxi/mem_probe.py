"""Phase-tagged VRAM probe for the OmniDreams perf pipeline on one GPU.

Samples driver-level used memory (torch.cuda.mem_get_info) and the torch
allocator's reserved bytes at 10 Hz on a background thread, so allocations made
outside the torch allocator (native extension cudaMalloc, cuDNN/cuBLAS/CUTLASS
workspaces, CUDA graph exec memory) show up as ``nontorch``. Prints a line
whenever driver-used memory moves by more than 256 MiB, tagged with the
current pipeline phase.

usage: python mem_probe.py [--width 1168] [--height 640] [--attn auto|sage3_fp8]
                           [--window 6] [--steps 4] [--skip-finalize 1]
"""

import argparse
import threading
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--width", type=int, default=1168)
ap.add_argument("--height", type=int, default=640)
ap.add_argument("--attn", default="auto")
ap.add_argument("--window", type=int, default=6)
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--native", default="required")
ap.add_argument("--skip-finalize", type=int, default=1)
ap.add_argument(
    "--empty-cache-steps",
    type=int,
    default=0,
    help="call torch.cuda.empty_cache() after each of the first N steps",
)
args = ap.parse_args()

phase = "import"
stop = False
GIB = 2**30
rows = []


def sampler():
    last = None
    while not stop:
        try:
            free, total = torch.cuda.mem_get_info()
            used = total - free
            reserved = torch.cuda.memory_reserved()
            alloc = torch.cuda.memory_allocated()
        except Exception:
            time.sleep(0.1)
            continue
        rows.append((time.perf_counter(), phase, used, reserved, alloc))
        if last is None or abs(used - last) > 256 * 2**20:
            print(
                f"[mem] {phase:28s} used={used / GIB:6.2f} torch_reserved={reserved / GIB:6.2f} "
                f"torch_alloc={alloc / GIB:6.2f} nontorch={(used - reserved) / GIB:6.2f} GiB",
                flush=True,
            )
            last = used
        time.sleep(0.1)


torch.cuda.init()
t = threading.Thread(target=sampler, daemon=True)
t.start()

# Imported after the sampler thread starts so the import phase is attributed.
from flashdreams.infra.config import derive_config  # noqa: E402
from omnidreams.config import OMNIDREAMS_PERF_PIPELINE_CONFIG  # noqa: E402

phase = "setup"
cfg = derive_config(
    OMNIDREAMS_PERF_PIPELINE_CONFIG,
    name="omnidreams-perf-memprobe",
    text_encoder=None,
    image_encoder=None,
    synthetic_text_max_length=512,
    diffusion_model={
        "transformer": {
            "native_dit_acceleration": args.native,
            "native_dit_attention_backend": args.attn,
            "window_size_t": args.window,
            "skip_finalize_kv_cache": bool(args.skip_finalize),
        }
    },
)
pipeline = cfg.setup()
phase = "to_cuda"
pipeline = pipeline.to("cuda").eval()
torch.cuda.synchronize()

net = cfg.diffusion_model.transformer.network
dec = pipeline.decoder
sc = int(dec.spatial_compression_ratio)
lh, lw = args.height // sc, args.width // sc
text_dim = (
    int(net.crossattn_proj_in_channels)
    if net.use_crossattn_projection
    else int(net.crossattn_emb_channels)
)
dtype = cfg.diffusion_model.transformer.dtype
phase = "init_cache"
cache = pipeline.initialize_cache_from_embeddings(
    text_embeddings=torch.zeros((1, 1, 512, text_dim), device="cuda", dtype=dtype),
    image_embeddings=torch.zeros(
        (1, 1, 1, int(net.in_channels), lh, lw), device="cuda", dtype=dtype
    ),
)
torch.cuda.synchronize()
gen = torch.Generator(device="cuda").manual_seed(0)
with torch.inference_mode():
    for i in range(args.steps):
        n = pipeline.get_num_output_frames(i)
        hdmap = (
            torch.rand(
                (1, 1, n, 3, args.height, args.width),
                generator=gen,
                device="cuda",
                dtype=dtype,
            )
            .mul_(2)
            .sub_(1)
        )
        phase = f"generate_{i}"
        t0 = time.perf_counter()
        video = pipeline.generate(autoregressive_index=i, cache=cache, input=hdmap)
        torch.cuda.synchronize()
        g = time.perf_counter() - t0
        phase = f"finalize_{i}"
        t0 = time.perf_counter()
        pipeline.finalize(autoregressive_index=i, cache=cache)
        torch.cuda.synchronize()
        print(
            f"[step {i}] frames={n} generate={g * 1000:.0f}ms finalize={(time.perf_counter() - t0) * 1000:.0f}ms "
            f"torch_peak={torch.cuda.max_memory_allocated() / GIB:.2f}",
            flush=True,
        )
        del video, hdmap
        if i < args.empty_cache_steps:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(
                f"[step {i}] empty_cache -> reserved={torch.cuda.memory_reserved() / GIB:.2f}",
                flush=True,
            )
phase = "done"
time.sleep(0.5)
stop = True
t.join(timeout=2)
import collections  # noqa: E402

peak = collections.defaultdict(lambda: [0, 0, 0])
for _, ph, used, res, al in rows:
    p = peak[ph]
    p[0] = max(p[0], used)
    p[1] = max(p[1], used - res)
    p[2] = max(p[2], res)
print(
    "phase                        peak_used  peak_nontorch  peak_torch_reserved (GiB)"
)
for ph, (u, nt, r) in peak.items():
    print(f"{ph:28s} {u / GIB:8.2f} {nt / GIB:12.2f} {r / GIB:14.2f}")
