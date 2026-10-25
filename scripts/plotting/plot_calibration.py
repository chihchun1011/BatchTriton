"""
Plot the E1 calibration sweep written by scripts/loadtest/calibrate.sh.

Usage:
    python plot_calibration.py ../../output/loadtest/calibration-20261016-120803

Writes <run_dir>/calibration.png with three panels against concurrency:
throughput (req/s), latency split into queue vs compute plus the client p99,
and GPU utilisation against the cobald low_utilisation threshold. Also prints
the table and the suggested C (throughput at the knee).

calibrate.sh repeats the sweep (a Run column in calibration.csv). Every point
is the mean over runs, with error bars of ± one standard deviation between
runs. A CSV without a Run column is treated as a single run (no error bars).
"""

import argparse
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from plot_tardis import LOW_UTIL  # noqa: E402

# Knee = lowest concurrency reaching this fraction of the peak throughput.
KNEE_FRACTION = 0.95

# Paper style: dark text on a white background, ticks inside on all sides.
BLUE, RED, ORANGE, GREEN = "#0072b2", "#d55e00", "#e69f00", "#009e73"  # Okabe-Ito
PAPER_STYLE = {
    "font.size": 10,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "axes.titlesize": 11,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "legend.fontsize": 8,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.fancybox": False,
}


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
        rows = list(csv.DictReader(f))

    batch = meta["batch"]
    conc = sorted({int(r["Concurrency"]) for r in rows})
    by_conc = {c: [r for r in rows if int(r["Concurrency"]) == c] for c in conc}
    n_runs = min(len(v) for v in by_conc.values())

    def stat(value):
        """value(row) for every run → (mean, std) per concurrency."""
        per_conc = [[value(r) for r in by_conc[c]] for c in conc]
        mean = np.array([np.mean(v) for v in per_conc])
        std = np.array([np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in per_conc])
        return mean, std

    def compute_us(r):
        return (float(r["Server Compute Input"]) + float(r["Server Compute Infer"])
                + float(r["Server Compute Output"]))

    req_s, req_s_sd = stat(lambda r: float(r["Inferences/Second"]) / batch)
    p50, _ = stat(lambda r: float(r["p50 latency"]) / 1000)
    p99, p99_sd = stat(lambda r: float(r["p99 latency"]) / 1000)
    queue, _ = stat(lambda r: float(r["Server Queue"]) / 1000)
    compute, _ = stat(lambda r: compute_us(r) / 1000)
    _, server_sd = stat(lambda r: (float(r["Server Queue"]) + compute_us(r)) / 1000)
    gpu, gpu_sd = stat(lambda r: gpu_value(r["Avg GPU Utilization"]))

    peak = req_s.max()
    knee = next(i for i, v in enumerate(req_s) if v >= KNEE_FRACTION * peak)

    print(f"{n_runs} run(s); mean ± std between runs")
    print(f"{'conc':>4} {'req/s':>13} {'p50 ms':>7} {'p99 ms':>12} "
          f"{'queue ms':>8} {'compute ms':>10} {'gpu util':>11}")
    for i in range(len(conc)):
        print(f"{conc[i]:4d} {req_s[i]:7.0f} ± {req_s_sd[i]:<3.0f} {p50[i]:7.1f} "
              f"{p99[i]:6.1f} ± {p99_sd[i]:<3.1f} {queue[i]:8.1f} {compute[i]:10.1f} "
              f"{gpu[i]:5.2f} ± {gpu_sd[i]:.2f}")
    print(f"\nPeak {peak:.0f} req/s; knee at concurrency {conc[knee]}: "
          f"C ≈ {req_s[knee]:.0f} ± {req_s_sd[knee]:.0f} req/s (batch {batch}), "
          f"GPU util {gpu[knee]:.2f} ± {gpu_sd[knee]:.2f}")

    with plt.rc_context(PAPER_STYLE):
        fig, (ax_t, ax_l, ax_g) = plt.subplots(1, 3, figsize=(15, 4.5))

        err = dict(capsize=3, elinewidth=0.8, capthick=0.8)

        ax_t.errorbar(conc, req_s, yerr=req_s_sd, color=BLUE, marker="o", **err)
        ax_t.axvline(conc[knee], color=GREEN, linestyle=":",
                     label=f"knee: C ≈ {req_s[knee]:.0f} req/s")
        ax_t.set_ylabel("requests / s")
        ax_t.set_title("Throughput")
        ax_t.legend(loc="lower right")

        # Bars on the log2 axis: same visual width at every concurrency.
        left = [c * 2 ** -0.3 for c in conc]
        width = [c * (2 ** 0.3 - 2 ** -0.3) for c in conc]
        ax_l.bar(left, compute, width=width, align="edge", color=BLUE,
                 label="server compute", zorder=2)
        ax_l.bar(left, queue, width=width, align="edge", bottom=compute, color=RED,
                 label="server queue", zorder=2)
        # Error bar on the stacked total (queue + compute), centred on each bar.
        ax_l.errorbar(conc, compute + queue, yerr=server_sd, fmt="none",
                      ecolor="black", zorder=3, **err)
        ax_l.errorbar(conc, p99, yerr=p99_sd, color="black", marker="o",
                      linestyle="--", label="client p99", zorder=4, **err)
        ax_l.set_ylabel("latency (ms)")
        ax_l.set_title("Latency per request")
        ax_l.legend(loc="upper left")

        ax_g.errorbar(conc, gpu, yerr=gpu_sd, color=ORANGE, marker="o",
                      label="nv_gpu_utilization", **err)
        ax_g.axhline(LOW_UTIL, color=RED, linestyle=":",
                     label=f"low_utilisation ({LOW_UTIL})")
        ax_g.set_ylim(0, 1.05)
        ax_g.set_ylabel("fraction")
        ax_g.set_title("GPU utilisation")
        ax_g.legend(loc="lower right")

        for ax in (ax_t, ax_l, ax_g):
            ax.set_xscale("log", base=2)
            ax.set_xticks(conc, [str(c) for c in conc])
            ax.xaxis.set_minor_locator(LogLocator(base=2, subs=(1.25, 1.5, 1.75)))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.grid(which="major", color="#b0b0b0", linewidth=0.6)
            ax.grid(which="minor", color="#dddddd", linewidth=0.4, linestyle=":")
            ax.set_axisbelow(True)
            ax.set_xlabel("concurrent requests")

        spread = f", mean ± std over {n_runs} runs" if n_runs > 1 else ""
        fig.suptitle(f"Calibration: {meta['drone_uuid']}  ({meta['model']}, batch {batch}, "
                     f"pf {meta['pf']}, sv {meta['sv']}{spread})", fontsize=13)
        fig.tight_layout()
        out = os.path.join(args.run_dir, "calibration.png")
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()
