"""
Pull the Triton server-side metrics for one load-test run out of Prometheus.

Reads the run window from <run_dir>/meta.json (written by loadgen.py), pads it
by --pad seconds on both sides, and writes <run_dir>/prometheus.csv in long
format: metric, drone_uuid, t, value.

Also cross-checks the client view against the server view: successful rows in
requests.csv vs the increase of nv_inference_request_success per drone.

Usage:
    python fetch_prometheus.py ../../output/loadtest/e2_step-20261016-120000
    python fetch_prometheus.py <run_dir> --prom http://localhost:9090

Prometheus URL: --prom, else $PROM_URL, else the ClusterIP of the
monitoring/prometheus service (via kubectl), else http://localhost:9090
(e.g. `kubectl -n monitoring port-forward svc/prometheus 9090:9090`).
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter

import requests

# {metric name in the CSV: PromQL}. {m} is replaced by the model selector.
# rate() windows are 30s = 2 scrape intervals (scrape_interval is 15s in
# k8s/prometheus/configmap.yaml), the shortest window that reliably has two
# samples to compute a rate from.
QUERIES = {
    "gpu_utilization": "nv_gpu_utilization",
    "up": 'up{job="triton"}',
    "success_total": "sum by (drone_uuid) (nv_inference_request_success{m})",
    "success_rate": "sum by (drone_uuid) (rate(nv_inference_request_success{m}[30s]))",
    "failure_rate": "sum by (drone_uuid) (rate(nv_inference_request_failure{m}[30s]))",
    "pending": "sum by (drone_uuid) (nv_inference_pending_request_count{m})",
    # Per-request averages: Δ(cumulative µs) / Δ(requests) over the window.
    "queue_us_per_req": (
        "sum by (drone_uuid) (rate(nv_inference_queue_duration_us{m}[30s]))"
        " / sum by (drone_uuid) (rate(nv_inference_request_success{m}[30s]))"
    ),
    "compute_us_per_req": (
        "sum by (drone_uuid) (rate(nv_inference_compute_infer_duration_us{m}[30s]))"
        " / sum by (drone_uuid) (rate(nv_inference_request_success{m}[30s]))"
    ),
    # Inferences per execution = how much dynamic batching actually happens.
    "batch_per_exec": (
        "sum by (drone_uuid) (rate(nv_inference_count{m}[30s]))"
        " / sum by (drone_uuid) (rate(nv_inference_exec_count{m}[30s]))"
    ),
}


def prometheus_url(arg: str | None) -> str:
    if arg:
        return arg
    if os.environ.get("PROM_URL"):
        return os.environ["PROM_URL"]
    try:
        ip = subprocess.run(
            ["kubectl", "-n", "monitoring", "get", "svc", "prometheus",
             "-o", "jsonpath={.spec.clusterIP}"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        if ip:
            return f"http://{ip}:9090"
    except (OSError, subprocess.SubprocessError):
        pass
    return "http://localhost:9090"


def query_range(prom: str, promql: str, start: float, end: float, step: float):
    r = requests.get(f"{prom}/api/v1/query_range", timeout=30, params={
        "query": promql, "start": start, "end": end, "step": step,
    })
    r.raise_for_status()
    body = r.json()
    if body["status"] != "success":
        raise RuntimeError(f"{promql}: {body.get('error')}")
    return body["data"]["result"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--prom", help="Prometheus base URL")
    p.add_argument("--step", type=float, default=5.0, help="query resolution (s)")
    p.add_argument("--pad", type=float, default=120.0,
                   help="seconds before/after the run to include")
    args = p.parse_args()

    with open(os.path.join(args.run_dir, "meta.json")) as f:
        meta = json.load(f)
    start = meta["start"] - args.pad
    end = meta.get("end", meta["start"] + meta["duration_s"]) + args.pad
    model_sel = '{model="%s"}' % meta["model"]
    prom = prometheus_url(args.prom)
    print(f"Querying {prom} for {end - start:.0f} s around the run")

    out_path = os.path.join(args.run_dir, "prometheus.csv")
    success_increase = {}
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "drone_uuid", "t", "value"])
        for name, promql in QUERIES.items():
            promql = promql.replace("{m}", model_sel)
            try:
                series = query_range(prom, promql, start, end, args.step)
            except Exception as e:
                print(f"warning: {name}: {e}", file=sys.stderr)
                continue
            for s in series:
                uuid = s["metric"].get("drone_uuid", "")
                for t, v in s["values"]:
                    writer.writerow([name, uuid, t, v])
                # Only the run itself, not the padding, plus one scrape
                # interval of slack before and two after for scrape lag.
                in_run = [float(v) for t, v in s["values"]
                          if meta["start"] - 15 <= t <= end - args.pad + 30]
                if name == "success_total" and in_run:
                    # Ignores counter resets - fine as long as a drone isn't
                    # restarted under the same uuid, which TARDIS never does.
                    success_increase[uuid] = in_run[-1] - in_run[0]
    print(f"Wrote {out_path}")

    requests_csv = os.path.join(args.run_dir, "requests.csv")
    if os.path.exists(requests_csv):
        with open(requests_csv) as f:
            client_ok = Counter(row["drone_uuid"] for row in csv.DictReader(f)
                                if row["status"] == "ok")
        print("\nClient ok vs server success increase (should match):")
        for uuid in sorted(set(client_ok) | set(success_increase)):
            server = success_increase.get(uuid)
            server_s = f"{server:.0f}" if server is not None else "n/a"
            print(f"  {uuid:24s} client={client_ok.get(uuid, 0):6d}  server={server_s}")


if __name__ == "__main__":
    main()
