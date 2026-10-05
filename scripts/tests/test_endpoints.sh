#!/usr/bin/env bash
# Smoke-test every live Triton endpoint listed in the TARDIS discovery file
# (configuration/endpoints.json, written by TritonAdapter._export_discovery_file).
#
# Usage:
#   ./test_endpoints.sh                     # uses ../../output/endpoints.json
#   ./test_endpoints.sh /path/to/other.json # or point at a different file
#
# For each entry, checks:
#   - /v2/health/ready on the HTTP port  (expect 200 if Triton is up)
#   - /metrics on the metrics port, grepped for nv_gpu_utilization
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENDPOINTS_FILE="${1:-$SCRIPT_DIR/../../output/endpoints.json}"

for bin in jq curl; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "error: '$bin' is required but not installed" >&2
    exit 1
  fi
done

if [[ ! -s "$ENDPOINTS_FILE" ]]; then
  echo "error: no endpoints found at $ENDPOINTS_FILE (missing or empty)" >&2
  exit 1
fi

echo "Testing Triton endpoints from $ENDPOINTS_FILE"
echo

while IFS= read -r entry; do
  drone_uuid=$(jq -r '.drone_uuid' <<<"$entry")
  host=$(jq -r '.host' <<<"$entry")
  http_port=$(jq -r '.http_port' <<<"$entry")
  metrics_port=$(jq -r '.metrics_port' <<<"$entry")

  echo "== $drone_uuid ($host) =="

  ready_code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
    "http://$host:$http_port/v2/health/ready" || echo "000")
  echo "  ready   (http=$http_port):     $ready_code"

  gpu_util=$(curl -s --max-time 5 "http://$host:$metrics_port/metrics" \
    | grep '^nv_gpu_utilization' || true)
  if [[ -n "$gpu_util" ]]; then
    echo "  metrics (metrics=$metrics_port):"
    echo "$gpu_util" | sed 's/^/    /'
  else
    echo "  metrics (metrics=$metrics_port):  unreachable or no nv_gpu_utilization sample"
  fi
  echo
done < <(jq -c '.[]' "$ENDPOINTS_FILE")
