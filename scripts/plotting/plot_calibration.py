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

    with plt.rc_context(PAPER_STYLE):
        fig, (ax_t, ax_l, ax_g) = plt.subplots(1, 3, figsize=(15, 4.5))

        ax_t.plot(conc, req_s, color=BLUE, marker="o")
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
        ax_l.plot(conc, p99, color="black", marker="o", label="client p99", zorder=3)
        ax_l.set_ylabel("latency (ms)")
        ax_l.set_title("Latency per request")
        ax_l.legend(loc="upper left")

        ax_g.plot(conc, gpu, color=ORANGE, marker="o", label="nv_gpu_utilization")
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

        fig.suptitle(f"Calibration: {meta['drone_uuid']}  ({meta['model']}, batch {batch}, "
                     f"pf {meta['pf']}, sv {meta['sv']})", fontsize=13)
        fig.tight_layout()
        out = os.path.join(args.run_dir, "calibration.png")
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()
