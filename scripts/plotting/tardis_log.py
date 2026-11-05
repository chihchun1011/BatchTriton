"""
Shared parsing for the plotting scripts: the COBalD config (controller
thresholds, cores per drone) and output/tardis.log (control-loop lines and
drone state transitions).
"""

import json
import os
import re
from collections import defaultdict
from datetime import datetime

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
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
# States after which the drone is gone; not carried into a cropped window.
GONE_STATES = {"Disintegrate", "ShutDown", "ShuttingDown", "Cleanup", "Down"}


def parse(source):
    """
    {
      "time", "demand", "utilisation", "allocation": control-loop lines,
      "supply_time", "supply": cores supplied, changing on drone state lines,
      "available": {drone_uuid: [Available since, until or None]},
      "states":    {drone_uuid: [(time, state), ...]}, state without "State",
    }
    Timestamps are naive local time, as TARDIS writes them.
    """
    data = defaultdict(list)
    supplying = {}  # drone_uuid -> cores, while the drone supplies
    data["available"] = {}
    data["states"] = defaultdict(list)
    for line in source:
        line = line.rstrip("\n")

        m = DRONE_RE.search(line)
        if m:
            ts, site, machine_type, uuid, state = m.groups()
            t = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            states = data["states"][uuid]
            if not states or states[-1][1] != state:
                states.append((t, state))
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


def read_log(path=TARDIS_LOG):
    with open(path, errors="ignore") as f:
        return parse(f)


def crop(data, lo, hi):
    """
    parse() output restricted to [lo, hi]. Supply and drone states carry
    their level from before lo into the window, and supply holds until hi.
    """
    out = {"available": data["available"]}

    keep = [i for i, t in enumerate(data["time"]) if lo <= t <= hi]
    for k in ("time", "demand", "utilisation", "allocation"):
        out[k] = [data[k][i] for i in keep]

    st, sv = data["supply_time"], data["supply"]
    before = [v for t, v in zip(st, sv) if t < lo]
    pairs = ([(lo, before[-1])] if before else []) + \
            [(t, v) for t, v in zip(st, sv) if lo <= t <= hi]
    if pairs:
        pairs.append((hi, pairs[-1][1]))
    out["supply_time"] = [t for t, _ in pairs]
    out["supply"] = [v for _, v in pairs]

    out["states"] = {}
    for uuid, transitions in data["states"].items():
        before = [s for t, s in transitions if t < lo]
        inside = [(t, s) for t, s in transitions if lo <= t <= hi]
        if before and before[-1] not in GONE_STATES:
            inside.insert(0, (lo, before[-1]))
        if inside:
            out["states"][uuid] = inside
    return out


def run_window(run_dir, pad=60.0):
    """(meta, start, end) of a load-test run, padded and as local datetimes."""
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)
    return (meta,
            datetime.fromtimestamp(meta["start"] - pad),
            datetime.fromtimestamp(meta["end"] + pad))
