#!/usr/bin/env bash
# E1 calibration: capacity curve of ONE Triton drone with perf_analyzer.
#
# Sweeps closed-loop concurrency and records throughput, latency percentiles
# and GPU utilisation, to find C (requests/s one drone sustains) - the `scale`
# in profiles/*.yaml - and the GPU utilisation at the latency knee.
#
# Usage:
#   ./calibrate.sh                   # first drone in ../../output/endpoints.json
#   ./calibrate.sh tritongpu-xxxx    # a specific drone
#
# Stop TARDIS after the drone is up so nothing scales during the sweep.
#
# The sweep is repeated RUNS times (run1/ … runN/) so the plot can show the
# run-to-run spread. MEASUREMENT_MS is the length of ONE perf_analyzer
# window: it repeats windows until 3 agree within 10 %, so a step takes
# about 3-4 windows. 30 s windows average ~15 samples of
# nv_gpu_utilization (Triton refreshes it every 2 s) instead of ~5.
#
# perf_analyzer reports Inferences/Second = requests/s × BATCH, so
#   C [req/s] = Inferences/Second / BATCH
# at the chosen operating point.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENDPOINTS_FILE="$SCRIPT_DIR/../../output/endpoints.json"
OUT_DIR="$(realpath -m "$SCRIPT_DIR/../../output/loadtest/calibration-$(date +%Y%m%d-%H%M%S)")"
SDK_IMAGE="docker://nvcr.io/nvidia/tritonserver:24.11-py3-sdk"   # same release as the server

MODEL="particlenet_AK4_PT"
# Max of the loadgen profiles: batch 1-40 jets, pf 20-50, sv 1-5.
BATCH=20
PF=50
SV=5
CONCURRENCIES=(1 2 4 8 16 32 64)
MEASUREMENT_MS=30000
RUNS=5

DRONE="${1:-}"
if [[ -n "$DRONE" ]]; then
  entry=$(jq -c --arg d "$DRONE" '.[] | select(.drone_uuid == $d)' "$ENDPOINTS_FILE")
else
  entry=$(jq -c '.[0]' "$ENDPOINTS_FILE")
fi
if [[ -z "$entry" || "$entry" == "null" ]]; then
  echo "error: drone not found in $ENDPOINTS_FILE" >&2
  exit 1
fi
DRONE=$(jq -r '.drone_uuid' <<<"$entry")
HOST=$(jq -r '.host' <<<"$entry")
# Ports come from the discovery file only.
GRPC_PORT=$(jq -r '.grpc_port' <<<"$entry")
METRICS_PORT=$(jq -r '.metrics_port' <<<"$entry")

mkdir -p "$OUT_DIR"
echo "Calibrating $DRONE ($HOST:$GRPC_PORT) → $OUT_DIR"
cat >"$OUT_DIR/meta.json" <<EOF
{"drone_uuid": "$DRONE", "model": "$MODEL", "batch": $BATCH, "pf": $PF, "sv": $SV, "runs": $RUNS, "measurement_ms": $MEASUREMENT_MS}
EOF

for r in $(seq 1 "$RUNS"); do
  RUN_DIR="$OUT_DIR/run$r"
  mkdir -p "$RUN_DIR"
  for c in "${CONCURRENCIES[@]}"; do
    echo "== run $r/$RUNS, concurrency $c"
    apptainer exec --bind "$OUT_DIR" "$SDK_IMAGE" perf_analyzer \
      -m "$MODEL" -i grpc -u "$HOST:$GRPC_PORT" \
      -b "$BATCH" \
      --shape "pf_points__0:2,$PF" --shape "pf_features__1:20,$PF" --shape "pf_mask__2:1,$PF" \
      --shape "sv_points__3:2,$SV" --shape "sv_features__4:11,$SV" --shape "sv_mask__5:1,$SV" \
      --concurrency-range "$c:$c" \
      --measurement-interval "$MEASUREMENT_MS" \
      --collect-metrics --metrics-url "http://$HOST:$METRICS_PORT/metrics" \
      --verbose-csv -f "$RUN_DIR/c$c.csv"
    # apptainer only binds $HOME, /tmp and $PWD by default, so without the
    # --bind above perf_analyzer's -f write silently goes nowhere.
    [[ -s "$RUN_DIR/c$c.csv" ]] || { echo "error: no CSV written for run $r, concurrency $c" >&2; exit 1; }
  done
done

# One CSV with a single header row and a leading Run column.
{
  printf 'Run,'; head -n 1 "$OUT_DIR/run1/c${CONCURRENCIES[0]}.csv"
  for r in $(seq 1 "$RUNS"); do
    for c in "${CONCURRENCIES[@]}"; do
      tail -n +2 "$OUT_DIR/run$r/c$c.csv" | sed "s/^/$r,/"
    done
  done
} >"$OUT_DIR/calibration.csv"
echo "Done → $OUT_DIR/calibration.csv"
echo "Plot:  python ../plotting/plot_calibration.py $OUT_DIR"
