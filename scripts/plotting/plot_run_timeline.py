"""
Load-test run timeline: lines up the load generator, Triton (via Prometheus)
and the COBalD/TARDIS control loop on one time axis.

Usage (from scripts/loadtest/):
    python ../plotting/plot_run_timeline.py ../../output/loadtest/e2_step-<ts>
    python ../plotting/plot_run_timeline.py <run_dir> --tardis-log ../../output/tardis.log

Needs in <run_dir>: requests.csv + meta.json (loadgen.py) and prometheus.csv
(fetch_prometheus.py). Writes into <run_dir>:
    timeline.png      rate / supply / utilisation / latency / queue time
    distribution.png  ok requests per drone over time
    drones.png        drone lifecycle (TARDIS states) + first request served
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "loadtest"))
from tardis_log import (DRONE_CORES, HIGH_ALLOC, LOW_UTIL, TARDIS_LOG,  # noqa: E402
                        crop, read_log, run_window)
from loadgen import Profile  # noqa: E402

BIN_S = 10
# For the "Tritons up × cores" line; assumes one machine type (gpu_pilot).
CORES_PER_DRONE = max(DRONE_CORES.values())

BG, FG, GRID = "#0f1117", "#aaaaaa", "#222222"
DRONE_COLORS = ["#4a9eff", "#ff6b6b", "#f7c948", "#7ed4a0", "#c792ea",
                "#ff9f43", "#5ce1e6", "#f78fb3", "#a3cb38", "#9aa5ce"]

STATE_COLORS = {
    "Request": "#555555", "Booting": "#ff9f43",
    "Integrate": "#f7c948", "Integrating": "#f7c948",
    "Available": "#7ed4a0", "Drain": "#ff6b6b",
    "Draining": "#ff6b6b", "Disintegrate": "#c792ea",
    "ShutDown": "#9aa5ce", "ShuttingDown": "#9aa5ce",
    "Cleanup": "#444444", "Down": "#333333",
}


def style(fig, axes):
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG)
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.grid(axis="y", color=GRID, linewidth=0.5)


def legend(ax, **kw):
    ax.legend(facecolor="#1a1a2e", edgecolor="#333333", labelcolor="white",
              fontsize=8, **kw)


def ts(epoch):
    # tardis.log timestamps are naive local time on this host, so convert the
    # epoch timestamps the same way to put both on one axis.
    return datetime.fromtimestamp(float(epoch))


def save(fig, run_dir, name):
    out = os.path.join(run_dir, name)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out}")


# ── Loading ───────────────────────────────────────────────────────────────────

def load_requests(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_prometheus(path):
    """{metric: {drone_uuid: (times[], values[])}}"""
    series = defaultdict(lambda: defaultdict(lambda: ([], [])))
    if not os.path.exists(path):
        print(f"warning: {path} missing - run fetch_prometheus.py first", file=sys.stderr)
        return series
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                v = float(row["value"])
            except ValueError:
                continue
            t, vals = series[row["metric"]][row["drone_uuid"]]
            t.append(float(row["t"]))
            vals.append(v)
    return series


def binned(times, edges, weights=None):
    counts, _ = np.histogram(times, bins=edges, weights=weights)
    return counts


def rolling_percentiles(times, values, edges, qs=(50, 99)):
    idx = np.digitize(times, edges) - 1
    out = {q: np.full(len(edges) - 1, np.nan) for q in qs}
    for b in range(len(edges) - 1):
        sel = values[idx == b]
        if len(sel):
            for q in qs:
                out[q][b] = np.percentile(sel, q)
    return out


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_timeline(run_dir, meta, reqs, prom, tardis, t0, t1):
    edges = np.arange(t0, t1 + BIN_S, BIN_S)
    centers = [ts(e + BIN_S / 2) for e in edges[:-1]]

    send_t = np.array([float(r["t_send"]) for r in reqs])
    sent = np.array([r["status"] in ("ok", "error") for r in reqs])
    ok = np.array([r["status"] == "ok" for r in reqs])
    err = np.array([r["status"] == "error" for r in reqs])
    lost = np.array([r["status"] in ("dropped", "no_endpoint") for r in reqs])
    latency = np.array([float(r["latency_ms"]) if r["latency_ms"] else np.nan for r in reqs])

    profile = Profile(meta["profile_cfg"], meta["scale"], meta["seed"])
    grid = np.linspace(meta["start"], meta["end"], 2000)
    offered = [profile.rate(t - meta["start"]) for t in grid]

    fig, axes = plt.subplots(5, 1, figsize=(13, 15), sharex=True,
                             gridspec_kw={"hspace": 0.08,
                                          "height_ratios": [1.2, 1.2, 1.2, 1, 1]})
    style(fig, axes)
    ax_rate, ax_sup, ax_util, ax_lat, ax_q = axes

    # 1. Offered vs achieved load
    ax_rate.plot([ts(t) for t in grid], offered, color="#aaaaaa", linewidth=1,
                 linestyle="--", label="Offered (profile)")
    ax_rate.plot(centers, binned(send_t[sent], edges) / BIN_S, color="#4a9eff",
                 linewidth=1.5, label="Sent")
    ax_rate.plot(centers, binned(send_t[ok], edges) / BIN_S, color="#7ed4a0",
                 linewidth=1.5, label="OK (client)")
    srv = prom.get("success_rate", {})
    if srv:
        tt = sorted({t for times, _ in srv.values() for t in times})
        total = defaultdict(float)
        for times, vals in srv.values():
            for t, v in zip(times, vals):
                total[t] += v
        ax_rate.plot([ts(t) for t in tt], [total[t] for t in tt], color="#c792ea",
                     linewidth=1, alpha=0.8, label="Success (Prometheus)")
    if lost.any():
        ax_rate.plot(centers, binned(send_t[lost], edges) / BIN_S, color="#ff6b6b",
                     linewidth=1, label="Dropped / no endpoint")
    ax_rate.set_ylabel("requests / s")
    ax_rate.set_title(f"Load test: {os.path.basename(run_dir)}  "
                      f"(model {meta['model']}, scale {meta['scale']})",
                      color="white", fontsize=13, pad=12)
    legend(ax_rate, loc="upper left")

    # 2. TARDIS demand / supply (cores) + ready Tritons
    if tardis["supply_time"]:
        ax_sup.step(tardis["supply_time"], tardis["supply"], where="post",
                    color="#4a9eff", linewidth=1.5, label="Supply (TARDIS)")
    if tardis["time"]:
        ax_sup.step(tardis["time"], tardis["demand"], where="post", color="#ff6b6b",
                    linewidth=1.5, label="Demand (TARDIS)")
    up = prom.get("up", {})
    if up:
        tt = sorted({t for times, _ in up.values() for t in times})
        n_up = defaultdict(float)
        for times, vals in up.values():
            for t, v in zip(times, vals):
                n_up[t] += v
        ax_sup.step([ts(t) for t in tt], [n_up[t] * CORES_PER_DRONE for t in tt],
                    where="post", color="#7ed4a0", linewidth=1.2,
                    label="Tritons scraped up (× cores)")
    ax_sup.set_ylabel("cpu core")
    right = ax_sup.secondary_yaxis("right", functions=(lambda c: c / CORES_PER_DRONE,
                                                        lambda d: d * CORES_PER_DRONE))
    right.set_ylabel("drones", color=FG)
    right.tick_params(colors=FG)
    legend(ax_sup, loc="upper left")

    # 3. Utilisation: per drone (Prometheus) + pool (what cobald acted on)
    gpu = prom.get("gpu_utilization", {})
    for i, (uuid, (times, vals)) in enumerate(sorted(gpu.items())):
        ax_util.plot([ts(t) for t in times], vals, linewidth=0.8, alpha=0.6,
                     color=DRONE_COLORS[i % len(DRONE_COLORS)], label=uuid)
    if tardis["time"]:
        ax_util.plot(tardis["time"], tardis["utilisation"], color="white",
                     linewidth=2, label="Pool utilisation (TARDIS)")
        ax_util.plot(tardis["time"], tardis["allocation"], color="#7ed4a0",
                     linewidth=1.2, linestyle="--", label="Pool allocation (TARDIS)")
    ax_util.axhline(LOW_UTIL, color="#ff6b6b", linewidth=0.8, linestyle=":",
                    label=f"low_utilisation ({LOW_UTIL})")
    ax_util.axhline(HIGH_ALLOC, color="#7ed4a0", linewidth=0.8, linestyle=":",
                    label=f"high_allocation ({HIGH_ALLOC})")
    ax_util.set_ylim(0, 1.05)
    ax_util.set_ylabel("fraction")
    legend(ax_util, loc="upper left", ncol=2)

    # 4. Latency + errors
    pct = rolling_percentiles(send_t[ok], latency[ok], edges)
    ax_lat.plot(centers, pct[50], color="#4a9eff", linewidth=1.5, label="p50")
    ax_lat.plot(centers, pct[99], color="#ff6b6b", linewidth=1.5, label="p99")
    ax_lat.set_yscale("log")
    ax_lat.set_ylabel("latency (ms)")
    ax_err = ax_lat.twinx()
    ax_err.bar(centers, binned(send_t[err], edges) / BIN_S,
               width=BIN_S / 86400, color="#ff6b6b", alpha=0.35, label="errors / s")
    ax_err.tick_params(colors=FG)
    ax_err.set_ylabel("errors / s", color=FG)
    lines = ax_lat.get_legend_handles_labels()
    bars = ax_err.get_legend_handles_labels()
    ax_lat.legend(lines[0] + bars[0], lines[1] + bars[1], facecolor="#1a1a2e",
                  edgecolor="#333333", labelcolor="white", fontsize=8, loc="upper left")

    # 5. Server-side queue time per request
    for i, (uuid, (times, vals)) in enumerate(sorted(prom.get("queue_us_per_req", {}).items())):
        ax_q.plot([ts(t) for t in times], np.array(vals) / 1000, linewidth=1,
                  color=DRONE_COLORS[i % len(DRONE_COLORS)], label=uuid)
    ax_q.set_ylabel("queue time (ms/req)")
    ax_q.set_xlabel("Time")
    if ax_q.lines:
        legend(ax_q, loc="upper left", ncol=2)

    for ax in axes:
        ax.axvspan(ts(meta["start"]), ts(meta["end"]), color="white", alpha=0.03)
    ax_q.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate(rotation=30, ha="right")
    save(fig, run_dir, "timeline.png")


def plot_distribution(run_dir, reqs, t0, t1):
    edges = np.arange(t0, t1 + BIN_S, BIN_S)
    centers = [ts(e + BIN_S / 2) for e in edges[:-1]]
    per_drone = defaultdict(list)
    for r in reqs:
        if r["status"] == "ok":
            per_drone[r["drone_uuid"]].append(float(r["t_send"]))
    if not per_drone:
        return
    uuids = sorted(per_drone)
    stacks = [binned(per_drone[u], edges) / BIN_S for u in uuids]

    fig, ax = plt.subplots(figsize=(13, 4))
    style(fig, [ax])
    ax.stackplot(centers, stacks, labels=uuids, alpha=0.85,
                 colors=[DRONE_COLORS[i % len(DRONE_COLORS)] for i in range(len(uuids))])
    ax.set_ylabel("ok requests / s")
    ax.set_title("Load distribution per drone", color="white", fontsize=13, pad=12)
    legend(ax, loc="upper left", ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate(rotation=30, ha="right")
    save(fig, run_dir, "distribution.png")


def plot_drones(run_dir, states, reqs, t1):
    """
    One row per drone: coloured spans for each TARDIS state, plus a marker at
    the first request the drone served. Request→Booting is the condor queue
    wait, Booting→Available is pilot start + Triton/model load, Available→
    first served is the load generator's endpoint refresh lag.
    """
    if not states:
        print("No drone state transitions in the run window - skipping drones.png")
        return
    # min, not first row: requests.csv is one block per loadgen worker.
    first_ok = {}
    for r in reqs:
        if r["status"] == "ok":
            t = float(r["t_send"])
            if r["drone_uuid"] not in first_ok or t < first_ok[r["drone_uuid"]]:
                first_ok[r["drone_uuid"]] = t
    first_ok = {u: ts(t) for u, t in first_ok.items()}

    uuids = sorted(states, key=lambda u: states[u][0][0])
    fig, ax = plt.subplots(figsize=(13, 0.5 * len(uuids) + 2))
    style(fig, [ax])
    seen = set()
    for row, uuid in enumerate(uuids):
        transitions = states[uuid] + [(ts(t1), None)]
        for (start, state), (end, _) in zip(transitions, transitions[1:]):
            ax.barh(row, mdates.date2num(end) - mdates.date2num(start),
                    left=mdates.date2num(start), height=0.6,
                    color=STATE_COLORS.get(state, "#888888"),
                    label=state if state not in seen else None)
            seen.add(state)
        if uuid in first_ok:
            ax.plot(mdates.date2num(first_ok[uuid]), row, marker="v", color="white",
                    markersize=7,
                    label="first request served" if "first" not in seen else None)
            seen.add("first")
    ax.set_yticks(range(len(uuids)))
    ax.set_yticklabels(uuids, color=FG, fontsize=8)
    ax.invert_yaxis()
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.set_title("Drone lifecycle (TARDIS states)", color="white", fontsize=13, pad=12)
    legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.autofmt_xdate(rotation=30, ha="right")
    save(fig, run_dir, "drones.png")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--tardis-log", default=TARDIS_LOG)
    p.add_argument("--pad", type=float, default=60.0,
                   help="seconds before/after the run to show")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    meta, lo, hi = run_window(args.run_dir, args.pad)
    t0, t1 = meta["start"] - args.pad, meta["end"] + args.pad
    reqs = load_requests(os.path.join(args.run_dir, "requests.csv"))
    prom = load_prometheus(os.path.join(args.run_dir, "prometheus.csv"))

    tardis = {"time": [], "demand": [], "utilisation": [], "allocation": [],
              "supply_time": [], "supply": [], "states": {}}
    if os.path.exists(args.tardis_log):
        tardis = crop(read_log(args.tardis_log), lo, hi)
    else:
        print(f"warning: {args.tardis_log} missing", file=sys.stderr)
    if not tardis["time"]:
        print("note: no TARDIS control-loop lines in the run window "
              "(was TARDIS running?)", file=sys.stderr)

    plot_timeline(args.run_dir, meta, reqs, prom, tardis, t0, t1)
    plot_distribution(args.run_dir, reqs, t0, t1)
    plot_drones(args.run_dir, tardis["states"], reqs, t1)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
