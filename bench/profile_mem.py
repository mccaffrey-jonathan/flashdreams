# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""OmniDreams VRAM allocation profiler.

Runs the pipeline in-process at a chosen config/resolution, monkeypatches the
lifecycle methods to record phase-delta GPU memory, and uses
``torch.cuda.memory._record_memory_history`` / ``_dump_snapshot`` to attribute
the steady-state live allocations to their call sites (grouped by module).

Goal: see *which groups of allocations* consume VRAM (weights vs CUDA-graph
pools vs KV cache vs CUTLASS/cuDNN workspace vs allocator fragmentation), and
whether each group is space-efficient for its tensor dimensions.

Usage (inside WSL via uv run):
    python profile_mem.py --tag native-1280 \
        --pixel-height 704 --pixel-width 1280 \
        --native required --compile-network False --use-cuda-graph True \
        --window-size-t 6 --steps 4
"""

from __future__ import annotations

import argparse
import collections
import gc
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

GIB = 1024**3
TIMELINE: list[dict] = []
GEN_CALLS: list[dict] = []
STEADY: dict = {}  # snapshot of memory stats captured DURING a steady generate


def gib(n: int | float) -> float:
    return round(n / GIB, 3)


def snap(label: str) -> float:
    import torch

    a = torch.cuda.memory_allocated()
    r = torch.cuda.memory_reserved()
    p = torch.cuda.max_memory_allocated()
    TIMELINE.append(
        {
            "label": label,
            "alloc_gib": gib(a),
            "reserved_gib": gib(r),
            "peak_gib": gib(p),
        }
    )
    return a


def parse_bool(s: str) -> bool:
    return str(s).lower() in ("1", "true", "yes", "on")


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument(
        "--config", default="omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf"
    )
    p.add_argument("--pixel-height", type=int, default=704)
    p.add_argument("--pixel-width", type=int, default=1280)
    p.add_argument(
        "--native", default="disabled", choices=["disabled", "auto", "required"]
    )
    p.add_argument("--compile-network", default="True")
    p.add_argument("--use-cuda-graph", default="True")
    p.add_argument("--vae-cuda-graph", default="True")
    p.add_argument("--window-size-t", type=int, default=None)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--uuid", default="239560dc-33d1-11ef-9720-00044bcbccac")
    p.add_argument(
        "--hdmap", default=None, help="explicit HDMap mp4 (skips full-clip preload)"
    )
    p.add_argument("--first-frame", default=None)
    p.add_argument("--outdir", default="/mnt/e/flashdreams/bench/mem")
    return p.parse_args()


def main() -> int:
    args = build_args()
    import torch
    from omnidreams import config as odc
    from omnidreams import pipeline as odp
    from omnidreams.runner import OmnidreamsRunnerConfig
    from flashdreams.infra.config import derive_config

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = Path(args.outdir) / f"{ts}_{args.tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    snap_path = outdir / "snapshot.pickle"

    # Record allocation stacks from the very start so resident weights are
    # attributed (not just allocations made after the first generate).
    try:
        torch.cuda.memory._record_memory_history(
            enabled="all", context="all", stacks="python", max_entries=150_000
        )
    except TypeError:  # older/newer signature
        torch.cuda.memory._record_memory_history(max_entries=150_000)

    base_pipe = odc.OMNIDREAMS_CONFIGS[args.config]
    tx = {
        "native_dit_acceleration": args.native,
        "compile_network": parse_bool(args.compile_network),
        "use_cuda_graph": parse_bool(args.use_cuda_graph),
    }
    if args.window_size_t is not None:
        tx["window_size_t"] = args.window_size_t
    vae_cg = parse_bool(args.vae_cuda_graph)
    pipe = derive_config(
        base_pipe,
        diffusion_model=dict(transformer=dict(**tx)),
        encoder=dict(use_cuda_graph=vae_cg),
        decoder=dict(use_cuda_graph=vae_cg),
        image_encoder=dict(use_cuda_graph=vae_cg),
    )
    rc_kw = dict(
        runner_name=f"memprofile-{args.tag}",
        pipeline=pipe,
        prompt=odc._DEFAULT_PROMPT_1V,
        output_dir=outdir,
        total_blocks=args.steps,
        pixel_height=args.pixel_height,
        pixel_width=args.pixel_width,
    )
    if args.hdmap:
        from pathlib import Path as _P

        rc_kw.update(
            hdmap_video_paths=(_P(args.hdmap),),
            first_frame_paths=(_P(args.first_frame),),
            example_data=False,
        )
    else:
        rc_kw.update(example_data=True, example_data_uuid=args.uuid)
    runner_cfg = OmnidreamsRunnerConfig(**rc_kw)

    # ---- monkeypatch lifecycle for phase-delta memory ----
    Pipe = odp.OmnidreamsPipeline
    orig_init_cache = Pipe.initialize_cache_from_embeddings
    orig_release = Pipe.release_oneshot_encoders
    orig_generate = Pipe.generate

    def wrap_init_cache(self, *a, **k):
        snap("before initialize_cache (KV alloc)")
        out = orig_init_cache(self, *a, **k)
        torch.cuda.synchronize()
        snap("after initialize_cache (KV alloc)")
        return out

    def wrap_release(self, *a, **k):
        snap("before release_oneshot_encoders")
        out = orig_release(self, *a, **k)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        snap("after release_oneshot_encoders (+empty_cache)")
        return out

    gen_n = {"i": 0}

    steady_idx = max(1, args.steps - 2)  # a settled step (graphs captured, KV warm)

    def wrap_generate(self, *a, **k):
        i = gen_n["i"]
        if i == 0:
            snap("before generate[0] (graph capture + workspace)")
        before = torch.cuda.memory_allocated()
        out = orig_generate(self, *a, **k)
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()
        GEN_CALLS.append(
            {
                "i": i,
                "alloc_before_gib": gib(before),
                "alloc_after_gib": gib(after),
                "peak_gib": gib(torch.cuda.max_memory_allocated()),
            }
        )
        if i == 0:
            snap("after generate[0]")
        if i == steady_idx:
            # capture the live working set DURING a steady generate (the number
            # that must fit in 32 GiB), before the loop tears down/.cpu()s chunks.
            try:
                torch.cuda.memory._dump_snapshot(str(snap_path))
            except Exception as exc:
                print(f"[profile] steady snapshot failed: {exc}")
            STEADY["stats"] = torch.cuda.memory_stats()
            snap(f"steady working set (generate[{i}])")
        gen_n["i"] += 1
        return out

    Pipe.initialize_cache_from_embeddings = wrap_init_cache
    Pipe.release_oneshot_encoders = wrap_release
    Pipe.generate = wrap_generate

    torch.cuda.reset_peak_memory_stats()
    snap("start (pre-build)")
    runner = runner_cfg.setup()
    torch.cuda.synchronize()
    snap("after setup() (weights resident)")

    # run the rollout (writes nothing useful for us; we want the memory timeline)
    try:
        runner.run()
    except Exception as exc:  # keep the snapshot even if late steps fail
        TIMELINE.append({"label": f"run() raised: {type(exc).__name__}: {exc}"})

    snap("post-run (chunks freed)")
    # Prefer the working-set snapshot/stats captured DURING a steady generate.
    stats = STEADY.get("stats") or torch.cuda.memory_stats()
    if not snap_path.exists():
        try:
            torch.cuda.memory._dump_snapshot(str(snap_path))
        except Exception as exc:
            print(f"[profile] snapshot dump failed: {exc}")
    report = analyze(snap_path, stats)
    out = {
        "ts": ts,
        "tag": args.tag,
        "config": args.config,
        "pixels": args.pixel_height * args.pixel_width,
        "res": f"{args.pixel_width}x{args.pixel_height}",
        "native": args.native,
        "compile_network": parse_bool(args.compile_network),
        "use_cuda_graph": parse_bool(args.use_cuda_graph),
        "vae_cuda_graph": vae_cg,
        "window_size_t": args.window_size_t,
        "timeline": TIMELINE,
        "generate_calls": GEN_CALLS,
        "alloc_gib": gib(stats.get("allocated_bytes.all.current", 0)),
        "reserved_gib": gib(stats.get("reserved_bytes.all.current", 0)),
        "active_gib": gib(stats.get("active_bytes.all.current", 0)),
        "inactive_split_gib": gib(stats.get("inactive_split_bytes.all.current", 0)),
        "num_alloc_retries": stats.get("num_alloc_retries", 0),
        "num_ooms": stats.get("num_ooms", 0),
        "groups": report,
    }
    (outdir / "mem_report.json").write_text(json.dumps(out, indent=2))
    ledger = Path(args.outdir) / "mem_ledger.jsonl"
    with open(ledger, "a") as f:
        f.write(json.dumps({k: out[k] for k in out if k not in ("timeline",)}) + "\n")
    print_summary(out)
    return 0


def _user_frame(frames: list[dict]) -> str:
    """Pick the most informative call-site for a block's allocation stack."""
    if not frames:
        return "<no-stack>"
    for fr in frames:
        fn = fr.get("filename", "")
        if (
            "site-packages/torch/" not in fn
            and "/torch/" not in fn
            and "<built-in>" not in fn
            and fn
        ):
            base = fn.rsplit("/", 1)[-1]
            return f"{base}:{fr.get('name', '?')}"
    fr = frames[0]
    base = fr.get("filename", "?").rsplit("/", 1)[-1]
    return f"{base}:{fr.get('name', '?')}"


def analyze(snap_path: Path, stats: dict) -> dict:
    """Group live allocations by call site and by segment pool type."""
    if not snap_path.exists():
        return {"error": "no snapshot"}
    snap = pickle.loads(snap_path.read_bytes())
    by_site: dict[str, list[int]] = collections.defaultdict(list)
    pool_bytes: dict[str, int] = collections.defaultdict(int)
    live_total = 0
    seg_reserved = 0
    for seg in snap.get("segments", []):
        seg_reserved += seg.get("total_size", 0)
        # CUDA-graph private pools are flagged via "is_expandable"/"segment_pool_id"
        pool_id = seg.get("segment_pool_id") or seg.get("owner_private_pool_id")
        pkey = (
            "graph_pool" if pool_id and any(pool_id) else seg.get("segment_type", "seg")
        )
        for blk in seg.get("blocks", []):
            if blk.get("state") in ("active_allocated", "active_pending_free"):
                sz = blk.get("size", 0) or blk.get("requested_size", 0)
                live_total += sz
                pool_bytes[pkey] += sz
                by_site[_user_frame(blk.get("frames", []))].append(sz)
    groups = sorted(
        (
            {
                "site": k,
                "gib": gib(sum(v)),
                "count": len(v),
                "avg_mib": round(sum(v) / len(v) / (1024**2), 1),
            }
            for k, v in by_site.items()
        ),
        key=lambda d: d["gib"],
        reverse=True,
    )[:25]
    return {
        "live_alloc_gib": gib(live_total),
        "segment_reserved_gib": gib(seg_reserved),
        "by_pool_gib": {k: gib(v) for k, v in pool_bytes.items()},
        "top_sites": groups,
    }


def print_summary(out: dict) -> None:
    print("\n" + "=" * 70)
    print(
        f"[mem] {out['tag']}  {out['res']} ({out['pixels']:,} px)  "
        f"native={out['native']} cuda_graph={out['use_cuda_graph']} "
        f"vae_cg={out['vae_cuda_graph']} wst={out['window_size_t']}"
    )
    print(
        f"  alloc={out['alloc_gib']}  reserved={out['reserved_gib']}  "
        f"active={out['active_gib']}  inactive_split(frag)={out['inactive_split_gib']}  "
        f"OOMs={out['num_ooms']} retries={out['num_alloc_retries']}"
    )
    g = out["groups"]
    if "top_sites" in g:
        print(f"  live={g['live_alloc_gib']}  by_pool={g['by_pool_gib']}")
        print("  --- phase deltas ---")
        prev = None
        for t in out["timeline"]:
            if "alloc_gib" not in t:
                print(f"    {t['label']}")
                continue
            d = "" if prev is None else f"  Δ{t['alloc_gib'] - prev:+.2f}"
            print(f"    {t['alloc_gib']:6.2f} GiB  {t['label']}{d}")
            prev = t["alloc_gib"]
        print("  --- top allocation sites ---")
        for s in g["top_sites"][:15]:
            print(
                f"    {s['gib']:6.2f} GiB  x{s['count']:<4} ({s['avg_mib']} MiB avg)  {s['site']}"
            )
    print("=" * 70, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
