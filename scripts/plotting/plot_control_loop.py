"""
COBalD/TARDIS control-loop plotter: demand vs supply, utilisation and
allocation against the controller thresholds, with controller holds shaded.

Usage (from scripts/loadtest/):
    # One load-test run: crop to its window, use the real GPU utilisation
    # from prometheus.csv if present, write <run_dir>/control_loop.png
    python ../plotting/plot_control_loop.py ../../output/loadtest/e2_step-<ts>

    # The whole log, no load test needed: output/tardis_plots/control_loop.png
    python ../plotting/plot_control_loop.py

Without prometheus.csv the utilisation panel shows the values from the log.
These have gaps: while the controller holds, TARDIS logs nothing, so there is
no GPU utilisation for those stretches.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from tardis_log import (HIGH_ALLOC, INTERVAL, LOW_UTIL, TARDIS_LOG,  # noqa: E402
                        crop, read_log, run_window)

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "output", "tardis_plots")

# A gap between demand lines longer than this means the controller held.
# Two intervals, since log timestamps jitter by a second or so.
HOLD_GAP = timedelta(seconds=2 * INTERVAL)


def load_gpu_utilisation(path, available):
    """
    nv_gpu_utilization from prometheus.csv (fetch_prometheus.py), kept only
    while a drone was Available - the drones the controller averages over.
    Returns ({drone_uuid: (times, values)}, mean_times, means).
    """
    def is_available(uuid, t):
        if uuid not in available:
            return False
        since, until = available[uuid]
        return since <= t and (until is None or t < until)

    per_drone = defaultdict(lambda: ([], []))
    by_time = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["metric"] != "gpu_utilization":
                continue
            # Same local-time conversion as the naive tardis.log timestamps.
            t = datetime.fromtimestamp(float(row["t"]))
            if not is_available(row["drone_uuid"], t):
                continue
            v = float(row["value"])
            times, values = per_drone[row["drone_uuid"]]
            times.append(t)
            values.append(v)
            by_time[t].append(v)
    mean_t = sorted(by_time)
    return per_drone, mean_t, [sum(by_time[t]) / len(by_time[t]) for t in mean_t]


def hold_gaps(times):
    """(start, end) of each stretch without demand lines, i.e. controller holding."""
    return [(a + timedelta(seconds=INTERVAL), b)
            for a, b in zip(times, times[1:]) if b - a > HOLD_GAP]


def with_gaps(times, values, gaps):
    """Insert a NaN at the start of each hold gap so the line breaks there."""
    out_t, out_v = list(times), list(values)
    for start, _ in reversed(gaps):
        i = next(i for i, t in enumerate(out_t) if t > start)
        out_t.insert(i, start)
        out_v.insert(i, float("nan"))
    return out_t, out_v


def plot(data, out, prom_csv=None, title="COBalD/TARDIS Control Loop", xlim=None):
    if not data["time"]:
        print("No control-loop lines found (wrong window, or check the log format).")
        return None

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08}
    )

    fig.patch.set_facecolor("#0f1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0f1117")
        ax.tick_params(colors="#aaaaaa")
        ax.xaxis.label.set_color("#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    t = data["time"]
    gaps = hold_gaps(t)
    for ax in (ax1, ax2):
        for i, (start, end) in enumerate(gaps):
            ax.axvspan(start, end, color="#ffffff", alpha=0.06, linewidth=0,
                       label="Hold" if i == 0 and ax is ax2 else None)

    # ── Top panel: demand vs supply ───────────────────────────────────────────
    # Supply from drone states: 0 before the first drone, held to the end.
    supply_t, supply = list(data["supply_time"]), list(data["supply"])
    if not supply_t or supply_t[0] > t[0]:
        supply_t.insert(0, t[0])
        supply.insert(0, 0)
    if supply_t[-1] < t[-1]:
        supply_t.append(t[-1])
        supply.append(supply[-1])
    ax1.step(supply_t, supply, where="post", color="#4a9eff", linewidth=1.5,
             label="Supply", alpha=0.9)
    ax1.step(t, data["demand"], where="post", color="#ff6b6b", linewidth=1.5,
             label="Demand", alpha=0.9)
    ax1.set_ylabel("cpu core", color="#aaaaaa", fontsize=11)
    ax1.legend(facecolor="#1a1a2e", edgecolor="#333333", labelcolor="white", fontsize=9)
    ax1.set_title(title, color="white", fontsize=13, pad=12)
    ax1.grid(axis="y", color="#222222", linewidth=0.5)

    # ── Bottom panel: utilisation + allocation ────────────────────────────────
    if prom_csv:
        # TritonAdapter reports the same GPU utilisation as allocation, so
        # one curve covers both.
        per_drone, mean_t, mean = load_gpu_utilisation(prom_csv, data["available"])
        for i, (uuid, (times, values)) in enumerate(sorted(per_drone.items())):
            ax2.plot(times, values, color="#aaaaaa", linewidth=0.8, alpha=0.5,
                     label="GPU util per drone" if i == 0 else None)
        ax2.plot(mean_t, mean, color="#f7c948", linewidth=1.5,
                 label="Mean over drones")
    else:
        ax2.step(*with_gaps(t, data["utilisation"], gaps), where="post",
                 color="#f7c948", linewidth=1.5, label="Utilisation")
        ax2.step(*with_gaps(t, data["allocation"], gaps), where="post",
                 color="#7ed4a0", linewidth=1.5, label="Allocation", linestyle="--")
    ax2.axhline(y=LOW_UTIL,  color="#ff6b6b", linewidth=0.8, linestyle=":", alpha=0.6, label=f"Low Utilization ({LOW_UTIL})")
    ax2.axhline(y=HIGH_ALLOC, color="#7ed4a0", linewidth=0.8, linestyle=":", alpha=0.6, label=f"High Allocation ({HIGH_ALLOC})")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Fraction", color="#aaaaaa", fontsize=11)
    ax2.set_xlabel("Time", color="#aaaaaa", fontsize=11)
    ax2.legend(facecolor="#1a1a2e", edgecolor="#333333", labelcolor="white", fontsize=9)
    ax2.grid(axis="y", color="#222222", linewidth=0.5)

    # ── Shared x-axis formatting ──────────────────────────────────────────────
    if xlim:
        ax2.set_xlim(*xlim)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate(rotation=30, ha="right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", nargs="?",
                   help="load-test run dir (meta.json, optionally prometheus.csv)")
    p.add_argument("--tardis-log", default=TARDIS_LOG)
    p.add_argument("--pad", type=float, default=60.0,
                   help="seconds before/after the run to show")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    data = read_log(args.tardis_log)

    if args.run_dir:
        _, lo, hi = run_window(args.run_dir, args.pad)
        data = crop(data, lo, hi)
        prom_csv = os.path.join(args.run_dir, "prometheus.csv")
        if not os.path.exists(prom_csv):
            print(f"note: {prom_csv} missing - using the log utilisation "
                  "(run fetch_prometheus.py for the real one)", file=sys.stderr)
            prom_csv = None
        fig = plot(data, os.path.join(args.run_dir, "control_loop.png"), prom_csv,
                   title=f"COBalD/TARDIS Control Loop: {os.path.basename(os.path.normpath(args.run_dir))}",
                   xlim=(lo, hi))
    else:
        fig = plot(data, os.path.join(OUTPUT_DIR, "control_loop.png"))

    if fig and args.show:
        plt.show()


if __name__ == "__main__":
    main()
