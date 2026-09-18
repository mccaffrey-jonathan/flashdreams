#!/usr/bin/env bash
# Headless Crazy Robotaxi fit/throughput probe (Linux or WSL2).
# usage: taxi_bench.sh TAG SLUG [run-v2 args...] -- [app args...]
# env:   FLASHDREAMS_ROOT (repo checkout to run from, default: this script's repo)
#        TAXI_BENCH_OUT   (where runs/<TAG> is written, default: <this dir>/runs)
#        TAXI_TIMEOUT     (seconds, default 2400)
set -u
TAG=$1; SLUG=$2; shift 2
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${FLASHDREAMS_ROOT:-$HERE/../..}"
# WSL2: libcuda.so lives outside the default linker search path (ludus plugin links -lcuda)
if [ -d /usr/lib/wsl/lib ]; then
  export LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}
fi
OUT=${TAXI_BENCH_OUT:-$HERE/runs}/$TAG
mkdir -p "$OUT"
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"; sleep 1; done ) > "$OUT/vram.log" 2>/dev/null &
SAMPLER=$!
echo "slug=$SLUG args=$*" > "$OUT/cmd.txt"
start=$(date +%s)
timeout ${TAXI_TIMEOUT:-2400} uv run --no-sync flashdreams-run-v2 "$SLUG" --mode mp4 --output-path "$OUT/out.mp4" \
  --backpressure-mode block --presentation-mode on_demand --stats-path "$OUT/stats.json" "$@" > "$OUT/run.log" 2>&1
rc=$?
end=$(date +%s)
kill $SAMPLER 2>/dev/null
echo "rc=$rc wall=$((end-start))s" | tee "$OUT/result.txt"
python3 "$HERE/parse_stats.py" "$OUT" | tee -a "$OUT/result.txt"
