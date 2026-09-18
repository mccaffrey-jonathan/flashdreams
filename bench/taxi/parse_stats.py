"""Summarize a taxi_bench run: steady-state timings from stats.json + peak VRAM.

Appends one JSON line per run to bench/taxi/ledger.jsonl.
"""

import json
import statistics
import sys
import time
from pathlib import Path

out = Path(sys.argv[1])
row = {"tag": out.name, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
cmd = out / "cmd.txt"
if cmd.exists():
    row["cmd"] = cmd.read_text().strip()
stats = out / "stats.json"
if stats.exists():
    d = json.loads(stats.read_text())
    frames = {s["step_index"]: s["frame_count"] for s in d["steps"]}
    by = {}
    for s in d["samples"]:
        by.setdefault(s["name"], {})[s["step_index"]] = s["value"]
    steady = sorted(i for i in frames if i >= 3)
    row["res"] = f"{d['session']['video_width']}x{d['session']['video_height']}"
    row["steps"] = len(frames)

    def med(name):
        vals = [by[name][i] for i in steady if name in by and i in by[name]]
        return statistics.median(vals) if vals else None

    fps_frames = frames[steady[-1]] if steady else 0
    for name, key in (
        ("total_s", "model_s"),
        ("diffuse_s", "diffuse_s"),
        ("encode_s", "encode_s"),
        ("decode_s", "decode_s"),
        ("model_step_wall_s", "step_wall_s"),
        ("engine_wall_s", "engine_s"),
        ("simulation_wall_s", "sim_s"),
        ("rules_wall_s", "rules_s"),
        ("conditioning_wall_s", "cond_s"),
        ("realtime_margin_s", "margin_s"),
        ("mem_peak_gib", "torch_peak_gib"),
        ("mem_reserved_gib", "torch_reserved_gib"),
    ):
        v = med(name)
        if v is not None:
            row[key] = round(v, 4)
    if row.get("model_s"):
        row["model_fps"] = round(fps_frames / row["model_s"], 1)
    if row.get("step_wall_s"):
        row["e2e_fps"] = round(fps_frames / row["step_wall_s"], 1)
vram = out / "vram.log"
if vram.exists():
    vals = [
        int(line.split()[1])
        for line in vram.read_text().splitlines()
        if len(line.split()) == 2 and line.split()[1].isdigit()
    ]
    if vals:
        row["smi_peak_mib"] = max(vals)
        tail = vals[len(vals) // 2 :]
        row["smi_steady_mib"] = int(statistics.median(tail))
log = out / "run.log"
if log.exists():
    txt = log.read_text(errors="replace")
    row["oom"] = txt.count("out of memory")
    row["traceback"] = txt.count("Traceback")
    row["rt_miss"] = txt.count("missed realtime budget")
print(json.dumps(row))
with (Path(__file__).parent / "ledger.jsonl").open("a") as f:
    f.write(json.dumps(row) + "\n")
