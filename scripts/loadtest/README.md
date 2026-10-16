# Load tests

Drive the TARDIS-managed Tritons with real inference requests and check that
COBalD/TARDIS scales on real `nv_gpu_utilization`. Before you run anything, remove
`simulated_utilisation_file` from `configuration/cobald_htc.yml`. Otherwise
TARDIS ignores the real metrics.

Model: `particlenet_AK4_PT`. The ONNX models in the CMSSW repository fail on the
V100 nodes (`cudaErrorNoKernelImageForDevice` with the `tritonserver:24.11`
onnxruntime). Only the `pytorch_libtorch` models (`*_PT`) work there.

## Run order

All commands are run from `scripts/loadtest/` with the `tardis-venv` python.

```bash
# E0: one request per drone, plus which GPU each drone reports
python loadgen.py --sanity

# E1: capacity of one drone (stop TARDIS once the drone is up)
./calibrate.sh [drone_uuid]
python ../plotting/plot_calibration.py ../../output/loadtest/calibration-<ts>
# Then set `scale:` in profiles/e*.yaml to C (req/s).

# E2-E4: scaling runs, with TARDIS running
python loadgen.py profiles/e2_step.yaml          # or --scale <C>
python fetch_prometheus.py ../../output/loadtest/e2_step-<ts>
python ../plotting/plot_loadtest.py ../../output/loadtest/e2_step-<ts>
```

`fetch_prometheus.py` finds Prometheus through the ClusterIP of
`monitoring/prometheus`. Use `--prom` or `$PROM_URL` to override it. It also
checks that the number of successful requests per drone matches between
`requests.csv` and Prometheus.

## Outputs (`output/loadtest/<run>/`)

| File | Content |
|---|---|
| `requests.csv` | One row per request: `ok`, `error`, `dropped` (client in-flight cap), or `no_endpoint` |
| `meta.json` | Profile, scale, seed, start/end time |
| `prometheus.csv` | GPU util, success/failure rate, queue/compute time per request, batch per exec, `up` |
| `timeline.png` | Offered vs achieved rate · TARDIS demand/supply · utilisation and thresholds · latency/errors · queue time |
| `distribution.png` | Successful requests per drone over time |
| `drones.png` | TARDIS state per drone, plus the first request each drone served |

## Notes

- The PyTorch model takes ~15 s to warm up: the first requests per drone run at
  ~1 s, then drop to ~15 ms. New drones show the same spike.
- The generator is open-loop: requests arrive as a Poisson process whose rate
  follows the profile, no matter how fast the servers answer. That way
  unserved demand shows up as rising latency and utilisation, not as a lower
  send rate.
