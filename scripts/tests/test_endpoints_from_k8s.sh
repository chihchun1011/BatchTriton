#!/usr/bin/env bash
# Same checks as test_endpoints.sh, but run from *inside* the k8s cluster -
# this is the connectivity test for "can a pod (Envoy, Prometheus) actually
# reach the HTCondor worker nodes Triton pilots run on". Spins up one
# ephemeral pod, curls every endpoint from configuration/endpoints.json, then
# deletes itself.
#
# Usage:
#   ./test_endpoints_from_k8s.sh                     # uses ../../output/endpoints.json
#   ./test_endpoints_from_k8s.sh /path/to/other.json
#
# Env vars:
#   KUBE_NAMESPACE  - namespace to run the pod in (default: current context's)
#   POD_NAME        - name for the ephemeral pod (default: triton-endpoint-test-<pid>)
#
# Note: `kubectl run --rm` only deletes the pod if the client stays attached
# until it finishes - when run non-interactively that can silently fail to
# happen, leaving the pod behind. A fixed pod name would then collide with
# the leftover on every later run ("AlreadyExists") before ever testing
# anything, so the default name includes $$ for uniqueness, and any pod
# with the chosen name is cleaned up defensively before each run anyway
# (covers the case POD_NAME is pinned to a fixed value via the env var).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENDPOINTS_FILE="${1:-$SCRIPT_DIR/../../output/endpoints.json}"
POD_NAME="${POD_NAME:-triton-endpoint-test-$$}"

for bin in jq kubectl; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "error: '$bin' is required but not installed" >&2
    exit 1
  fi
done

if [[ ! -s "$ENDPOINTS_FILE" ]]; then
  echo "error: no endpoints found at $ENDPOINTS_FILE (missing or empty)" >&2
  exit 1
fi

namespace_args=()
if [[ -n "${KUBE_NAMESPACE:-}" ]]; then
  namespace_args=(-n "$KUBE_NAMESPACE")
fi

# Build one shell script that curls every endpoint, to run inside a single
# ephemeral pod (curlimages/curl has no shell scripting beyond `sh`, so this
# is composed here rather than looping with repeated `kubectl run` calls).
inner_script="set -e"$'\n'
while IFS= read -r entry; do
  drone_uuid=$(jq -r '.drone_uuid' <<<"$entry")
  host=$(jq -r '.host' <<<"$entry")
  http_port=$(jq -r '.http_port' <<<"$entry")
  metrics_port=$(jq -r '.metrics_port' <<<"$entry")

  inner_script+="echo '== $drone_uuid ($host) ==' "$'\n'
  inner_script+="echo -n '  ready   (http=$http_port):     '; curl -s -o /dev/null -w '%{http_code}\n' --max-time 5 http://$host:$http_port/v2/health/ready || echo unreachable"$'\n'
  inner_script+="echo '  metrics (metrics=$metrics_port):'; curl -s --max-time 5 http://$host:$metrics_port/metrics | grep '^nv_gpu_utilization' | sed 's/^/    /' || echo '    unreachable or no sample'"$'\n'
  inner_script+="echo"$'\n'
done < <(jq -c '.[]' "$ENDPOINTS_FILE")

echo "Testing Triton endpoints from inside the cluster (pod: $POD_NAME)"
echo

kubectl delete pod "$POD_NAME" "${namespace_args[@]}" --ignore-not-found --wait=true >/dev/null

kubectl run "$POD_NAME" "${namespace_args[@]}" --rm -i --tty=false --restart=Never \
  --image=curlimages/curl --command -- sh -c "$inner_script"
