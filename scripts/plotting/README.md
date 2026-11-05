# Plotting

Plots for the TARDIS control loop and the load tests in `scripts/loadtest/`.
Run them from `scripts/loadtest/`. Each one writes its PNGs into the run dir it is given.

| Script | Input | Output | Use |
|---|---|---|---|
| `plot_control_loop.py [run_dir]` | `output/tardis.log`, `prometheus.csv` if present | `<run_dir>/control_loop.png` | Did TARDIS start the right number of drones? Demand vs supply, holds shaded, utilisation vs thresholds |
| `plot_run_timeline.py <run_dir>` | `requests.csv`, `meta.json`, `prometheus.csv`, `tardis.log` | `timeline.png`, `distribution.png`, `drones.png` | Debugging one run: client rate, latency and drops, per-drone load, drone lifecycle |
| `plot_calibration.py <run_dir>` | `calibration.csv` from `calibrate.sh` | `calibration.png` | E1: throughput, latency and GPU utilisation vs concurrency, and the suggested C |
| `tardis_log.py` | | | Shared module, not run directly: config thresholds, `tardis.log` parsing, run window |

```bash
python ../plotting/plot_control_loop.py ../../output/loadtest/e2_step-<ts>
python ../plotting/plot_run_timeline.py ../../output/loadtest/e2_step-<ts>
python ../plotting/plot_calibration.py  ../../output/loadtest/calibration-<ts>
```

Notes:
- Without a run dir, `plot_control_loop.py` plots the whole log to
  `output/tardis_plots/control_loop.png`. No load test is needed for this.
- `--tardis-log` and `--pad` (seconds shown before and after the run, default 60) work for both
  run plotters. `--show` opens a window.
- The thresholds and cores per drone come from `configuration/cobald_htc.yml`,
  so the plots follow the config.
- `tardis.log` timestamps are naive local time. Epoch timestamps are converted
  the same way, so run the plots on the host where TARDIS runs, or in the same timezone.
