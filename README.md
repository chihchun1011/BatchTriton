# SONIC Screwdriver

COBalD/TARDIS for ML inference serving: scale NVIDIA Triton Inference
Servers on opportunistic GPU resources up and down with the demand for
inference.

SONIC (Services for Optimized Network Inference on Coprocessors) is the CMS
approach to run ML inference as a service: CMSSW jobs send their inputs to a remote Triton
server instead of running the models in-process. That needs a pool of Triton
servers whose size follows the load. SONIC Screwdriver provides it with the
tools CMS already uses to integrate opportunistic resources:

- **TARDIS** submits *pilots* through HTCondor. Each pilot starts one Triton
  server in Apptainer on one GPU and serves the CMSSW models from CVMFS.
- A **TritonAdapter** (in the [`tardis-src`](tardis-src) submodule, branch
  `feature/triton-support`) lets TARDIS manage those servers. It checks
  readiness on Triton's health endpoint and reads GPU utilisation from
  Triton's metrics. It also writes the live endpoints to a discovery file
  for clients, Prometheus or a proxy.
- **COBalD**'s controller adds a pilot when the GPUs are busy and drains one
  when they idle.

```
CMSSW / clients ──gRPC──▶ Triton pilots (HTCondor + Apptainer, 1 GPU each)
       ▲                        │ health, GPU utilisation
       │ endpoints.json         ▼
       └────────────── TARDIS (TritonAdapter) ◀── COBalD controller
```

## Layout

| Path | Content |
|---|---|
| `configuration/` | COBalD/TARDIS config (`cobald_htc.yml`), pilot JDL and start script |
| `tardis-src/` | TARDIS fork with the TritonAdapter (git submodule) |
| `k8s/` | Prometheus that scrapes the drones through the discovery file |
| `scripts/loadtest/` | Load generator, calibration and the E0–E4 scaling tests ([README](scripts/loadtest/README.md)) |
| `scripts/plotting/` | Control-loop and load-test plots ([README](scripts/plotting/README.md)) |
| `scripts/tests/` | Endpoint checks |
| `output/` | Logs, drone registry, `endpoints.json`, run outputs (not tracked) |

## Status

A prototype on a GridKa test node (V100 GPUs) serving the CMSSW ParticleNet
AK4 model (`particlenet_AK4_PT`). Load tests show the pool scaling up to
3 drones under a 2.2 × single-GPU load step and back down after it.
