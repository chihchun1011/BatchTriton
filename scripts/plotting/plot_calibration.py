"""
Plot the E1 calibration sweep written by scripts/loadtest/calibrate.sh.

Usage:
    python plot_calibration.py ../../output/loadtest/calibration-20261016-120803

Writes <run_dir>/calibration.png with three panels against concurrency:
throughput (req/s), latency split into queue vs compute, and GPU utilisation
against the cobald low_utilisation threshold. Also prints the table and the
suggested C (throughput at the knee).
"""

import argparse
import csv
import json
import os
import sys

import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from plot_tardis import LOW_UTIL  # noqa: E402
from plot_loadtest import FG, legend, save, style  # noqa: E402

# Knee = lowest concurrency reaching this fraction of the peak throughput.
KNEE_FRACTION = 0.95


def gpu_value(cell):
    """'GPU-ab94...:0.334324;' (one entry per GPU) → mean value."""
    vals = [float(p.split(":")[1]) for p in cell.strip().split(";") if ":" in p]
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    args = p.parse_args()

    with open(os.path.join(args.run_dir, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(args.run_dir, "calibration.csv")) as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["Concurrency"]))

    batch = meta["batch"]
    conc = [int(r["Concurrency"]) for r in rows]
    req_s = [float(r["Inferences/Second"]) / batch for r in rows]
    p50 = [float(r["p50 latency"]) / 1000 for r in rows]
    p99 = [float(r["p99 latency"]) / 1000 for r in rows]
    queue = [float(r["Server Queue"]) / 1000 for r in rows]
    compute = [
        (float(r["Server Compute Input"]) + float(r["Server Compute Infer"])
         + float(r["Server Compute Output"])) / 1000 for r in rows]
    gpu = [gpu_value(r["Avg GPU Utilization"]) for r in rows]

    peak = max(req_s)
    knee = next(i for i, v in enumerate(req_s) if v >= KNEE_FRACTION * peak)

    print(f"{'conc':>4} {'req/s':>8} {'p50 ms':>7} {'p99 ms':>7} "
          f"{'queue ms':>8} {'compute ms':>10} {'gpu util':>8}")
    for i in range(len(rows)):
        print(f"{conc[i]:4d} {req_s[i]:8.0f} {p50[i]:7.1f} {p99[i]:7.1f} "
              f"{queue[i]:8.1f} {compute[i]:10.1f} {gpu[i]:8.2f}")
    print(f"\nPeak {peak:.0f} req/s; knee at concurrency {conc[knee]}: "
          f"C ≈ {req_s[knee]:.0f} req/s (batch {batch}), GPU util {gpu[knee]:.2f}")

    fig, (ax_t, ax_l, ax_g) = plt.subplots(1, 3, figsize=(15, 4.5))
    style(fig, [ax_t, ax_l, ax_g])

    ax_t.plot(conc, req_s, color="#4a9eff", marker="o", linewidth=1.5)
    ax_t.axvline(conc[knee], color="#7ed4a0", linestyle=":", linewidth=1,
                 label=f"knee: C ≈ {req_s[knee]:.0f} req/s")
    ax_t.set_ylabel("requests / s")
    ax_t.set_title("Throughput", color="white")
    legend(ax_t, loc="lower right")

    ax_l.bar([str(c) for c in conc], compute, color="#4a9eff", label="server compute")
    ax_l.bar([str(c) for c in conc], queue, bottom=compute, color="#ff6b6b",
             label="server queue")
    ax_l.plot([str(c) for c in conc], p99, color="white", marker="o", linewidth=1.2,
              label="client p99")
    ax_l.set_ylabel("latency (ms)")
    ax_l.set_title("Latency per request", color="white")
    legend(ax_l, loc="upper left")

    ax_g.plot(conc, gpu, color="#f7c948", marker="o", linewidth=1.5,
              label="nv_gpu_utilization")
    ax_g.axhline(LOW_UTIL, color="#ff6b6b", linestyle=":", linewidth=1,
                 label=f"low_utilisation ({LOW_UTIL})")
    ax_g.set_ylim(0, 1.05)
    ax_g.set_ylabel("fraction")
    ax_g.set_title("GPU utilisation", color="white")
    legend(ax_g, loc="lower right")

    for ax in (ax_t, ax_g):
        ax.set_xscale("log", base=2)
        ax.set_xticks(conc, [str(c) for c in conc])
    for ax in (ax_t, ax_l, ax_g):
        ax.set_xlabel("concurrent requests", color=FG)

    fig.suptitle(f"Calibration: {meta['drone_uuid']}  ({meta['model']}, batch {batch}, "
                 f"pf {meta['pf']}, sv {meta['sv']})", color="white", fontsize=13)
    fig.tight_layout()
    save(fig, args.run_dir, "calibration.png")


if __name__ == "__main__":
    main()
