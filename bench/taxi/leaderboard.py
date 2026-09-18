"""Render bench/taxi/LEADERBOARD.md from ledger.jsonl (latest entry per tag)."""

import json
from pathlib import Path

here = Path(__file__).parent
rows = {}
for line in (here / "ledger.jsonl").read_text().splitlines():
    if line.strip():
        d = json.loads(line)
        rows[d["tag"]] = d
cols = [
    ("tag", "run"),
    ("res", "res"),
    ("e2e_fps", "e2e fps"),
    ("model_fps", "model fps"),
    ("step_wall_s", "step wall s"),
    ("model_s", "model s"),
    ("diffuse_s", "diffuse s"),
    ("encode_s", "encode s"),
    ("engine_s", "engine s"),
    ("margin_s", "rt margin s"),
    ("smi_peak_mib", "smi peak MiB"),
    ("smi_steady_mib", "smi steady MiB"),
    ("torch_peak_gib", "torch peak GiB"),
    ("oom", "oom"),
    ("rt_miss", "rt miss"),
]
out = [
    "# Crazy Robotaxi on RTX 5090 (32 GiB) - headless WSL2 runs",
    "",
    "40 blocks, `--no-ui --profile-pipeline`, boulevard_district map, steady state = median over AR steps >= 3.",
    "e2e fps = frames per chunk / model-thread wall (engine + model); 30 fps needs step wall <= 0.267 s.",
    "",
    "| " + " | ".join(h for _, h in cols) + " |",
    "|" + "---|" * len(cols),
]


def fmt(v):
    return "" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


for d in sorted(rows.values(), key=lambda r: -(r.get("e2e_fps") or 0)):
    out.append("| " + " | ".join(fmt(d.get(k)) for k, _ in cols) + " |")
out += [
    "",
    "Commands per run are in `runs/<tag>/cmd.txt`; raw stats in `runs/<tag>/stats.json`.",
]
(here / "LEADERBOARD.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
