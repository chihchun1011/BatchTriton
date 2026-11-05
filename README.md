# SONIC Screwdriver

NVIDIA Triton Inference Servers on batch and opportunistic GPU resources,
scaled up and down with the demand for inference by
[COBalD/TARDIS](https://cobald-tardis.readthedocs.io/), which German HEP
already uses to integrate such resources. Clients send their inputs to these
servers instead of running the models in-process (inference as a service, see
[SONIC](https://github.com/fastmachinelearning/SonicCMS)).

- **TARDIS** submits *pilots* through HTCondor. Each pilot starts one Triton
  server in Apptainer on one GPU, with the models served from CVMFS.
- A **TritonAdapter** (in the [`tardis-src`](tardis-src) submodule, branch
  `feature/triton-support`) lets TARDIS manage those servers. It checks
  readiness on Triton's health endpoint and reads GPU utilisation from
  Triton's metrics. It also writes the live endpoints to a discovery file
  for clients, Prometheus or a proxy.
- **COBalD**'s controller adds a pilot when the GPUs are busy and drains one
  when they idle.

```
Clients ─────────gRPC──▶ Triton pilots (HTCondor + Apptainer, 1 GPU each)
       ▲                        │ health, GPU utilisation
       │ endpoints.json         ▼
       └────────────── TARDIS (TritonAdapter) ◀── COBalD controller
```

## Usage

### Requirements

- An HTCondor pool with GPU worker nodes that you can submit to.
- Apptainer with GPU support (`--nv`) and CVMFS on the worker nodes. The
  default pilot takes the Triton image from `unpacked.cern.ch` and the
  models from `cms.cern.ch`.
- Python 3.10 on the submit node.
- Optional: a Kubernetes cluster for the Prometheus monitoring in `k8s/`.

### 1. Install

```bash
git clone --recurse-submodules https://github.com/chihchun1011/sonic-screwdriver.git
cd sonic-screwdriver
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p output
```

`requirements.txt` installs the TARDIS fork from `tardis-src/` in editable
mode, so changes to the TritonAdapter take effect without reinstalling.

### 2. Configure for your site

| File | What to adapt |
|---|---|
| `configuration/cobald_htc.yml` | `pool` (your collector), site `quota`, `MachineMetaData` (cores, memory, disk per pilot), controller thresholds |
| `configuration/pilot.jdl` | GPU request options your site needs; `x509userproxy` if your pool requires a proxy |
| `configuration/start_pilot.sh` | Triton image and model repository |

If your pool needs an X.509 proxy, create one and point `x509userproxy` in
`pilot.jdl` at it:

```bash
voms-proxy-init --rfc --voms <your-vo> -valid 192:00
```

### 3. Run

```bash
cd configuration
python -m cobald.daemon cobald_htc.yml
```

TARDIS logs to `output/tardis.log` and writes the live Triton endpoints to
`output/endpoints.json`. Check that the servers answer with
`scripts/tests/test_endpoints.sh`.

### 4. Monitoring (optional)

```bash
kubectl apply -f k8s/namespace-monitoring.yaml
kubectl apply -f k8s/prometheus/
kubectl -n monitoring port-forward svc/prometheus 9090:9090
```

Set the `hostPath` in `k8s/prometheus/deployment.yaml` to your `output/`
directory first, since Prometheus finds the drones through `endpoints.json`.
To reach the UI from your laptop, forward the port over SSH:
`ssh -L 9090:localhost:9090 <node>`.

### Troubleshooting

- **Apptainer fails to create a writable layer with `--nv`:** set
  `use nvidia-container-cli = no` in `/etc/apptainer/apptainer.conf`.

To test scaling with real load, see [scripts/loadtest](scripts/loadtest/README.md).

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
