"""
COBalD/TARDIS Control Loop Plotter Usage:
    # From the log TARDIS writes:
    python3 ../plotting/plot_tardis.py

    # Real GPU utilisation from Prometheus
    # (run fetch_prometheus.py on the load-test run first):
    python3 ../plotting/plot_tardis.py --prometheus ../../output/loadtest/e2_step-<ts>

Without --prometheus the utilisation panel shows the values from the log
(tardis_plot.png). These have gaps: while the controller holds, TARDIS logs
nothing, so there is no GPU utilisation for those stretches.
"""

import argparse
import csv
import os
import re
import sys
import yaml
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "output", "tardis_plots")

COBALD_CONFIG = os.path.join(SCRIPT_DIR, "..", "..", "configuration", "cobald_htc.yml")
TARDIS_LOG = os.path.join(SCRIPT_DIR, "..", "..", "output", "tardis.log")


def load_config(path=COBALD_CONFIG):
    """
    LinearController thresholds and interval, and cores per drone, from the
    cobald config.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    controller = next(e for e in cfg["pipeline"]
                      if e.get("__type__", "").endswith("LinearController"))
    tardis = cfg["tardis"]
    cores = {
        (site["name"], machine_type): meta["Cores"]
        for site in tardis["Sites"]
        for machine_type, meta in tardis[site["name"]]["MachineMetaData"].items()
    }
    return (controller["low_utilisation"], controller["high_allocation"],
            controller.get("interval", 1), cores)


# Read rather than hard-coded, so the plotted lines can't drift from the config.
LOW_UTIL, HIGH_ALLOC, INTERVAL, DRONE_CORES = load_config()

# A gap between demand lines longer than this means the controller held.
# Two intervals, since log timestamps jitter by a second or so.
HOLD_GAP = timedelta(seconds=2 * INTERVAL)

# ── Regex ──────────────────────────────────────────────────────────────────────
# Matches lines like:
# changes: 2026-06-22 12:20:12 demand = 566 [demand=567, supply=640.0, utilisation=0.78, allocation=1.00]
LOG_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"   # timestamp
    r".*?"
    r"demand\s*=\s*([\d.]+)"                      # current demand
    r".*?"
    r"supply\s*=\s*([\d.]+)"                      # supply
    r".*?"
    r"utilisation\s*=\s*([\d.]+)"                 # utilisation
    r".*?"
    r"allocation\s*=\s*([\d.]+)"                  # allocation
)

# Drone state lines, e.g.
# cobald.runtime.tardis.resources.dronestates: 2026-10-25 22:08:51 Drone {'site_name': 'TritonGPU',
#   'machine_type': 'gpu_pilot', ..., 'drone_uuid': 'tritongpu-392486051e', ...} in AvailableState
DRONE_RE = re.compile(
    r"dronestates: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) Drone "
    r"\{'site_name': '([^']+)', 'machine_type': '([^']+)'.*"
    r"'drone_uuid': '([^']+)'.* in (\w+)State$"
)
# A drone supplies its cores from AvailableState until the controller drains
# it (TARDIS sets supply to 0 on the way to DrainState) or it dies.
SUPPLY_END_STATES = {"Drain", "Draining", "Disintegrate", "ShutDown",
                     "ShuttingDown", "Cleanup", "Down"}


def parse(source):
    data = defaultdict(list)
    supplying = {}  # drone_uuid -> cores, while the drone supplies
    # drone_uuid -> [Available since, until] (until None: still at log end)
    data["available"] = {}
    for line in source:
        line = line.rstrip("\n")

        m = DRONE_RE.search(line)
        if m:
            ts, site, machine_type, uuid, state = m.groups()
            t = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            if state == "Available" and uuid not in supplying:
                supplying[uuid] = DRONE_CORES[site, machine_type]
                data["available"][uuid] = [t, None]
            elif state in SUPPLY_END_STATES and uuid in supplying:
                del supplying[uuid]
                data["available"][uuid][1] = t
            else:
                continue
            data["supply_time"].append(t)
            data["supply"].append(sum(supplying.values()))
            continue

        m = LOG_RE.search(line)
        if not m:
            continue
        ts, demand, _, util, alloc = m.groups()
        # deduplicate: skip if same second already recorded
        t = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        if data["time"] and data["time"][-1] == t:
            continue
        data["time"].append(t)
        data["demand"].append(float(demand))
        data["utilisation"].append(float(util))
        data["allocation"].append(float(alloc))
    return data


def load_gpu_utilisation(run_dir, available):
    """
    nv_gpu_utilization from <run_dir>/prometheus.csv (fetch_prometheus.py),
    kept only while a drone was Available - the drones the controller
    averages over. Returns ({drone_uuid: (times, values)}, mean_times, means).
    """
    path = os.path.join(run_dir, "prometheus.csv")
    if not os.path.exists(path):
        sys.exit(f"error: {path} missing - run fetch_prometheus.py first")

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


def plot(data, prom_run=None):
    if not data["time"]:
        print("No matching log lines found. Check the log format.")
        return

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
    # Supply from drone states, extended to the last demand line.
    supply_t = [t[0]] + data["supply_time"] + [t[-1]]
    supply = [0] + data["supply"] + [data["supply"][-1] if data["supply"] else 0]
    ax1.step(supply_t, supply, where="post", color="#4a9eff", linewidth=1.5,
             label="Supply", alpha=0.9)
    ax1.step(t, data["demand"], where="post", color="#ff6b6b", linewidth=1.5,
             label="Demand", alpha=0.9)
    ax1.set_ylabel("cpu core", color="#aaaaaa", fontsize=11)
    ax1.legend(facecolor="#1a1a2e", edgecolor="#333333", labelcolor="white", fontsize=9)
    ax1.set_title("COBalD/TARDIS Control Loop", color="white", fontsize=13, pad=12)
    ax1.grid(axis="y", color="#222222", linewidth=0.5)

    # ── Bottom panel: utilisation + allocation ────────────────────────────────
    if prom_run:
        # TritonAdapter reports the same GPU utilisation as allocation, so
        # one curve covers both.
        per_drone, mean_t, mean = load_gpu_utilisation(prom_run, data["available"])
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
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate(rotation=30, ha="right")

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR,
                       "tardis_plot_prometheus.png" if prom_run else "tardis_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prometheus", metavar="RUN_DIR",
                   help="load-test run dir with prometheus.csv: plot real GPU utilisation")
    args = p.parse_args()

    with open(TARDIS_LOG, errors="ignore") as f:
        data = parse(f)

    plot(data, args.prometheus)
