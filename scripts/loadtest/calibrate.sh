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
# The first run pulls the tritonserver SDK image (several GB) into the
# apptainer cache.
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
# Mean of the loadgen profiles: batch 1-40 jets, pf 20-50, sv 1-5.
BATCH=20
PF=35
SV=3
CONCURRENCIES=(1 2 4 8 16 32 64)
MEASUREMENT_MS=10000

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
GRPC_PORT=$(( $(jq -r '.http_port' <<<"$entry") + 1 ))   # see start_pilot.sh
METRICS_PORT=$(jq -r '.metrics_port' <<<"$entry")

mkdir -p "$OUT_DIR"
echo "Calibrating $DRONE ($HOST:$GRPC_PORT) → $OUT_DIR"
cat >"$OUT_DIR/meta.json" <<EOF
{"drone_uuid": "$DRONE", "model": "$MODEL", "batch": $BATCH, "pf": $PF, "sv": $SV}
EOF

for c in "${CONCURRENCIES[@]}"; do
  echo "== concurrency $c"
  apptainer exec --bind "$OUT_DIR" "$SDK_IMAGE" perf_analyzer \
    -m "$MODEL" -i grpc -u "$HOST:$GRPC_PORT" \
    -b "$BATCH" \
    --shape "pf_points__0:2,$PF" --shape "pf_features__1:20,$PF" --shape "pf_mask__2:1,$PF" \
    --shape "sv_points__3:2,$SV" --shape "sv_features__4:11,$SV" --shape "sv_mask__5:1,$SV" \
    --concurrency-range "$c:$c" \
    --measurement-interval "$MEASUREMENT_MS" \
    --collect-metrics --metrics-url "http://$HOST:$METRICS_PORT/metrics" \
    --verbose-csv -f "$OUT_DIR/c$c.csv"
  # apptainer only binds $HOME, /tmp and $PWD by default, so without the
  # --bind above perf_analyzer's -f write silently goes nowhere.
  [[ -s "$OUT_DIR/c$c.csv" ]] || { echo "error: no CSV written for concurrency $c" >&2; exit 1; }
done

# One CSV with a single header row.
first=1
for c in "${CONCURRENCIES[@]}"; do
  if (( first )); then cat "$OUT_DIR/c$c.csv"; first=0; else tail -n +2 "$OUT_DIR/c$c.csv"; fi
done >"$OUT_DIR/calibration.csv"
echo "Done → $OUT_DIR/calibration.csv"
echo "Plot:  python ../plotting/plot_calibration.py $OUT_DIR"
