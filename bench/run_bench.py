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
"""OmniDreams 5090 benchmark driver — one offline-runner rollout, profiled & logged.

Wraps ``flashdreams-run <config> [tuning flags]``, parses the per-AR-step
``stats_<config>.json`` it emits, computes steady-state throughput (fps), and
appends a row to a JSONL ledger + regenerates a markdown leaderboard.

Designed to be driven repeatedly with different tuning flags (the optimize
loop). Runs *inside WSL* via ``uv run python``; writes the ledger to the
Windows side (/mnt/e/...) so it can be read directly from the host.

Usage (inside WSL, env already set up by the launcher):
    uv run --no-sync --package flashdreams-omnidreams python run_bench.py \
        --tag baseline-perf \
        --config omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf \
        --total-blocks 16 \
        -- \
        --pipeline.diffusion-model.transformer.native-dit-acceleration required
Everything after the bare ``--`` is forwarded verbatim to flashdreams-run.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_UUID = "239560dc-33d1-11ef-9720-00044bcbccac"
NUM_FRAMES_RE = re.compile(r"AR step (\d+)/\d+, num_frames=(\d+)")
OOM_RE = re.compile(
    r"CUDA out of memory|OutOfMemoryError|CUDA driver error: out of memory"
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="short human label for this run")
    p.add_argument("--config", required=True, help="runner slug")
    p.add_argument("--total-blocks", type=int, default=16)
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=2,
        help="leading AR steps excluded from steady-state median",
    )
    p.add_argument("--uuid", default=DEFAULT_UUID)
    p.add_argument(
        "--hdmap",
        default=None,
        help="explicit HDMap mp4 (avoids full-clip GPU preload artifact)",
    )
    p.add_argument("--first-frame", default=None)
    p.add_argument("--out-root", default=None, help="dir for mp4/stats (WSL ext4)")
    p.add_argument("--ledger", default="/mnt/e/flashdreams/bench/ledger.jsonl")
    p.add_argument("--timeout", type=int, default=3600)
    if "--" in sys.argv:
        i = sys.argv.index("--")
        own, passthru = sys.argv[1:i], sys.argv[i + 1 :]
    else:
        own, passthru = sys.argv[1:], []
    return p.parse_args(own), passthru


def run(args: argparse.Namespace, passthru: list[str]) -> dict:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(args.out_root) if args.out_root else Path.home() / "bench-runs"
    out_dir = out_root / f"{ts}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    cmd = [
        "flashdreams-run",
        args.config,
        "--total-blocks",
        str(args.total_blocks),
        "--output-dir",
        str(out_dir),
    ]
    if args.hdmap:
        cmd += [
            "--hdmap-video-paths",
            args.hdmap,
            "--first-frame-paths",
            args.first_frame,
        ]
    else:
        cmd += ["--example-data", "True", "--example_data_uuid", args.uuid]
    cmd += passthru
    print(f"[bench] {args.tag}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    log_lines: list[str] = []
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                log_lines.append(line)
                logf.write(line)
                logf.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            rc = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = -9
    wall_s = time.time() - t0
    log_text = "".join(log_lines)

    # per-step frames from the runner's log
    frames_by_step = {
        int(m.group(1)): int(m.group(2)) for m in NUM_FRAMES_RE.finditer(log_text)
    }

    stats_path = out_dir / f"stats_{args.config}.json"
    row: dict = {
        "ts": ts,
        "tag": args.tag,
        "config": args.config,
        "flags": " ".join(passthru),
        "total_blocks": args.total_blocks,
        "exit_code": rc,
        "wall_s": round(wall_s, 1),
        "log": str(log_path),
    }

    if rc != 0 or not stats_path.exists():
        reason = "OOM" if OOM_RE.search(log_text) else f"exit={rc}/no-stats"
        tail = "".join(log_lines[-12:]).strip()
        row.update({"status": "fail", "reason": reason, "fps": 0.0, "tail": tail})
        return row

    stats = json.loads(stats_path.read_text())
    steady = [s for s in stats if s["autoregressive_index"] >= args.warmup_steps]
    if not steady:
        steady = stats[-1:]

    def med(key: str) -> float:
        return round(statistics.median(s[key] for s in steady), 2)

    # steady frames/block (constant for index>=1); fall back to 8
    fb_vals = [frames_by_step.get(s["autoregressive_index"], 0) for s in steady]
    fb_vals = [v for v in fb_vals if v > 0]
    frames_per_block = fb_vals[0] if fb_vals else 8

    med_total = med("total_ms")
    med_total_wo = med("total_ms_wo_finalize")
    fps = round(frames_per_block * 1000.0 / med_total, 2) if med_total else 0.0
    fps_wo = round(frames_per_block * 1000.0 / med_total_wo, 2) if med_total_wo else 0.0
    peak = round(max(s["mem_peak_gib"] for s in stats), 2)

    row.update(
        {
            "status": "ok",
            "frames_per_block": frames_per_block,
            "n_steady": len(steady),
            "fps": fps,
            "fps_wo_finalize": fps_wo,
            "ms_total": med_total,
            "ms_total_wo_finalize": med_total_wo,
            "ms_diffuse": med("diffuse_ms"),
            "ms_encode": med("encode_ms"),
            "ms_decode": med("decode_ms"),
            "ms_finalize": med("finalize_ms"),
            "peak_gib": peak,
            "mp4": str(out_dir / f"{args.config}.mp4"),
        }
    )
    return row


def write_leaderboard(ledger: Path) -> None:
    rows = [json.loads(ln) for ln in ledger.read_text().splitlines() if ln.strip()]
    ok = sorted(
        [r for r in rows if r.get("status") == "ok"],
        key=lambda r: r["fps"],
        reverse=True,
    )
    md = [
        "# OmniDreams 5090 — two-step inference leaderboard",
        "",
        "Target: **30 fps** steady-state (8 frames/block => <=267 ms/block).",
        "",
        f"_Updated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} — "
        f"{len(ok)} ok / {len(rows)} total runs._",
        "",
        "| fps | fps(no-fin) | tag | diffuse | enc | dec | final | total ms | "
        "peak GiB | flags |",
        "|----:|----:|:--|----:|----:|----:|----:|----:|----:|:--|",
    ]
    for r in ok[:25]:
        md.append(
            f"| **{r['fps']}** | {r.get('fps_wo_finalize', '')} | {r['tag']} | "
            f"{r['ms_diffuse']} | {r['ms_encode']} | {r['ms_decode']} | "
            f"{r['ms_finalize']} | {r['ms_total']} | {r['peak_gib']} | "
            f"`{r['flags'] or '(perf defaults)'}` |"
        )
    fails = [r for r in rows if r.get("status") == "fail"]
    if fails:
        md += ["", "## Failed runs", "", "| tag | reason | flags |", "|:--|:--|:--|"]
        for r in fails[-15:]:
            md.append(f"| {r['tag']} | {r.get('reason', '')} | `{r['flags']}` |")
    (ledger.parent / "LEADERBOARD.md").write_text("\n".join(md) + "\n")


def main() -> int:
    args, passthru = parse_args()
    row = run(args, passthru)
    ledger = Path(args.ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a") as f:
        f.write(json.dumps(row) + "\n")
    write_leaderboard(ledger)

    print("\n" + "=" * 60)
    if row["status"] == "ok":
        print(
            f"[bench] {row['tag']}: {row['fps']} fps  "
            f"(no-finalize {row['fps_wo_finalize']} fps)"
        )
        print(
            f"        diffuse={row['ms_diffuse']}  encode={row['ms_encode']}  "
            f"decode={row['ms_decode']}  finalize={row['ms_finalize']}  "
            f"total={row['ms_total']} ms/block"
        )
        print(
            f"        frames/block={row['frames_per_block']}  "
            f"peak={row['peak_gib']} GiB  wall={row['wall_s']}s"
        )
    else:
        print(f"[bench] {row['tag']}: FAILED ({row['reason']}) after {row['wall_s']}s")
        print(row.get("tail", ""))
    print("=" * 60, flush=True)
    return 0 if row["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
