"""
COBalD/TARDIS Log Plotter
Usage:
    # Pipe live docker logs:
    docker logs -f <container_id> 2>&1 | python3 plot_tardis.py

    # Or from a saved log file:
    python3 plot_tardis.py tardis.log
"""

import os
import re
import sys
import yaml
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from collections import defaultdict

# Written next to this script's location rather than the current working
# directory, so the plot always lands in output/plots/ regardless of where
# this is run from (repo root, scripts/plotting/, piped from a log viewer, etc).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "output", "plots")

COBALD_CONFIG = os.path.join(SCRIPT_DIR, "..", "..", "configuration", "cobald_htc.yml")


def load_thresholds(path=COBALD_CONFIG):
    """low_utilisation / high_allocation of the LinearController in the cobald config."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    controller = next(e for e in cfg["pipeline"]
                      if e.get("__type__", "").endswith("LinearController"))
    return controller["low_utilisation"], controller["high_allocation"]


# Read rather than hard-coded, so the plotted lines can't drift from the config.
LOW_UTIL, HIGH_ALLOC = load_thresholds()

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


def parse(source):
    data = defaultdict(list)
    for line in source:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="ignore")
        m = LOG_RE.search(line)
        if not m:
            continue
        ts, demand, supply, util, alloc = m.groups()
        # deduplicate: skip if same second already recorded
        t = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        if data["time"] and data["time"][-1] == t:
            continue
        data["time"].append(t)
        data["demand"].append(float(demand))
        data["supply"].append(float(supply))
        data["utilisation"].append(float(util))
        data["allocation"].append(float(alloc))
    return data


def plot(data):
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

    # ── Top panel: demand vs supply ───────────────────────────────────────────
    ax1.plot(t, data["supply"],  color="#4a9eff", linewidth=1.5, label="Supply",  alpha=0.9)
    ax1.plot(t, data["demand"],  color="#ff6b6b", linewidth=1.5, label="Demand",  alpha=0.9)
    ax1.fill_between(t, data["demand"], data["supply"],
                     where=[d > s for d, s in zip(data["demand"], data["supply"])],
                     color="#ff6b6b", alpha=0.15, label="Demand > Supply")
    ax1.fill_between(t, data["demand"], data["supply"],
                     where=[d <= s for d, s in zip(data["demand"], data["supply"])],
                     color="#4a9eff", alpha=0.10, label="Supply ≥ Demand")
    ax1.set_ylabel("cpu core", color="#aaaaaa", fontsize=11)
    ax1.legend(facecolor="#1a1a2e", edgecolor="#333333", labelcolor="white", fontsize=9)
    ax1.set_title("COBalD/TARDIS Control Loop", color="white", fontsize=13, pad=12)
    ax1.grid(axis="y", color="#222222", linewidth=0.5)

    # ── Bottom panel: utilisation + allocation ────────────────────────────────
    ax2.plot(t, data["utilisation"], color="#f7c948", linewidth=1.5, label="Utilisation")
    ax2.plot(t, data["allocation"],  color="#7ed4a0", linewidth=1.5, label="Allocation",  linestyle="--")
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
    out = os.path.join(OUTPUT_DIR, "tardis_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            data = parse(f)
    else:
        # read from stdin (piped docker logs)
        data = parse(sys.stdin.buffer)
    
    # data selection (for .log)
    # N = 300
    # data = {k: v[-N:] for k, v in data.items()}
    plot(data)