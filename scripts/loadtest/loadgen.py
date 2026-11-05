"""
Open-loop load generator for the TARDIS-managed Triton pilots.

Requests arrive as a Poisson process whose rate follows a profile (see
profiles/*.yaml), independent of how fast the servers answer. That is the
point: with a closed loop (fixed concurrency) adding servers just dilutes the
load, so there is never unmet demand for TARDIS to scale against.

Endpoints are taken from the TARDIS discovery file (output/endpoints.json,
written by TritonAdapter._export_discovery_file) and re-read periodically.
Requests are spread round-robin over the endpoints that report ready.

One Python process tops out at ~1100 req/s (payload creation + gRPC on one
core), so the load is split over --workers processes, each sending a Poisson stream 
at rate/N.

Usage:
    python loadgen.py profiles/e2_step.yaml
    python loadgen.py profiles/e2_step.yaml --scale 1300   # multiply all rates
    python loadgen.py profiles/e2_step.yaml --workers 8   # override process count
    python loadgen.py --sanity                            # one request per endpoint (E0)

Each run writes to output/loadtest/<profile>-<timestamp>/:
    requests.csv  one row per request (sent, failed or dropped)
    meta.json     profile, start/end time - used by fetch_prometheus.py
"""

import argparse
import asyncio
import csv
import itertools
import json
import math
import multiprocessing as mp
import os
import queue
import shutil
import random
import sys
import time
from datetime import datetime

import aiohttp
import numpy as np
import tritonclient.grpc.aio as grpcclient
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_ENDPOINTS = os.path.join(REPO_DIR, "output", "endpoints.json")
OUTPUT_DIR = os.path.join(REPO_DIR, "output", "loadtest")

ENDPOINT_REFRESH_S = 5
REQUEST_TIMEOUT_S = 30
# Clients of drones that vanished from endpoints.json are closed only after
# this grace period, so requests still in flight to them finish (or fail) on
# their own instead of being cancelled by us.
CLIENT_CLOSE_GRACE_S = 60
# Rate one worker process is given at most. A single process measured
# ~1100 req/s at full load with p50 ~330 ms of client-side queueing; at
# 650 req/s it kept up (p50 12 ms), so stay below that.
WORKER_MAX_RATE = 600


# ── Load profile ──────────────────────────────────────────────────────────────

class Profile:
    """
    Piecewise request-rate profile (req/s) as a function of seconds since start.

    Phase kinds (all take `duration` in seconds):
      rate: 10              constant
      rate: [5, 40]         linear ramp from 5 to 40
      burst: {every, length, factor}   optional on top of rate: multiply the
                            rate by `factor` for `length` s every `every` s
      random_walk: {start, min, max, sigma, step}
                            rate changes by N(0, sigma) every `step` s,
                            clipped to [min, max]
    All rates are multiplied by the top-level `scale` (or --scale), so a
    profile can be written in units of one drone's capacity and rescaled after
    calibration.
    """

    def __init__(self, cfg: dict, scale: float, seed: int):
        self.scale = scale
        self.phases = cfg["phases"]
        self.duration = sum(p["duration"] for p in self.phases)
        self._rng = random.Random(seed)
        self._walk = {}  # phase index -> precomputed list of rates per step

    def rate(self, t: float) -> float:
        start = 0.0
        for i, phase in enumerate(self.phases):
            if t < start + phase["duration"]:
                return max(0.0, self._phase_rate(i, phase, t - start)) * self.scale
            start += phase["duration"]
        return 0.0

    def peak(self, step: float = 0.1) -> float:
        return max(self.rate(i * step) for i in range(int(self.duration / step) + 1))

    def _phase_rate(self, i: int, phase: dict, t: float) -> float:
        if "random_walk" in phase:
            w = phase["random_walk"]
            if i not in self._walk:
                steps = math.ceil(phase["duration"] / w["step"]) + 1
                rates, r = [], w["start"]
                for _ in range(steps):
                    rates.append(r)
                    r = min(w["max"], max(w["min"], r + self._rng.gauss(0, w["sigma"])))
                self._walk[i] = rates
            base = self._walk[i][int(t // w["step"])]
        elif isinstance(phase["rate"], list):
            r0, r1 = phase["rate"]
            base = r0 + (r1 - r0) * t / phase["duration"]
        else:
            base = phase["rate"]

        burst = phase.get("burst")
        if burst and (t % burst["every"]) < burst["length"]:
            base *= burst["factor"]
        return base


# ── Request payloads ──────────────────────────────────────────────────────────

class PayloadFactory:
    """
    Random input tensors with the shapes the model expects. 
    """

    def __init__(self, model_config, cfg: dict, rng: np.random.Generator):
        self.inputs = [(i.name, list(i.dims)) for i in model_config.input]
        self.outputs = [o.name for o in model_config.output]
        self.batch = cfg.get("batch", {"min": 1, "max": 1})
        self.shapes = cfg.get("shapes", {})
        self.rng = rng

    def make(self):
        batch = int(self.rng.integers(self.batch["min"], self.batch["max"] + 1))
        lengths = {
            prefix: int(self.rng.integers(lo, hi + 1))
            for prefix, (lo, hi) in self.shapes.items()
        }
        inputs = []
        for name, dims in self.inputs:
            prefix = name.split("_")[0]
            # example "pf_points": dims=[-1, 2]->[batch, shape, 2], "pf_features", dims=[-1, 16]->[batch, shape, 16]
            shape = [batch] + [lengths.get(prefix, 1) if d == -1 else d for d in dims]
            if "_mask" in name:
                data = np.ones(shape, dtype=np.float32)
            else:
                data = self.rng.standard_normal(shape, dtype=np.float32)
            tensor = grpcclient.InferInput(name, shape, "FP32")
            tensor.set_data_from_numpy(data)
            inputs.append(tensor)
        outputs = [grpcclient.InferRequestedOutput(n) for n in self.outputs]
        return inputs, outputs, batch, lengths


# ── Endpoints ─────────────────────────────────────────────────────────────────

class EndpointPool:
    """
    Tracks the ready Triton endpoints listed in the discovery file.

    Host and ports come only from the discovery file, never derived here.
    An endpoint is only used once /v2/health/ready answers 200: drones appear
    in endpoints.json as soon as their condor job runs, which can be well
    before Triton has finished loading models.
    """

    def __init__(self, path: str):
        self.path = path
        self.ready: list[dict] = []
        self._clients: dict[str, grpcclient.InferenceServerClient] = {}
        self._retired: list[tuple[float, grpcclient.InferenceServerClient]] = []
        self._rr = itertools.count()

    def client(self, ep: dict) -> grpcclient.InferenceServerClient:
        uuid = ep["drone_uuid"]
        if uuid not in self._clients:
            self._clients[uuid] = grpcclient.InferenceServerClient(
                url=f"{ep['host']}:{ep['grpc_port']}"
            )
        return self._clients[uuid]

    def next(self) -> dict | None:
        if not self.ready:
            return None
        return self.ready[next(self._rr) % len(self.ready)]

    def read_listed(self) -> list[dict]:
        """
        All entries of the discovery file. An unreadable file counts as no
        drones (e.g. before TARDIS's first condor_q refresh has written it).
        """
        try:
            with open(self.path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"warning: could not read {self.path}: {e}", file=sys.stderr)
            return []

    async def refresh(self, session: aiohttp.ClientSession) -> None:
        listed = self.read_listed()

        async def is_ready(ep):
            url = f"http://{ep['host']}:{ep['http_port']}/v2/health/ready"
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as r:
                    return r.status == 200
            except Exception:
                return False

        flags = await asyncio.gather(*(is_ready(ep) for ep in listed))
        self.ready = [ep for ep, ok in zip(listed, flags) if ok]

        now = time.monotonic()
        alive = {ep["drone_uuid"] for ep in self.ready}
        for uuid in [u for u in self._clients if u not in alive]:
            self._retired.append((now, self._clients.pop(uuid)))
        while self._retired and now - self._retired[0][0] > CLIENT_CLOSE_GRACE_S:
            await self._retired.pop(0)[1].close()

    async def close(self) -> None:
        for c in list(self._clients.values()) + [c for _, c in self._retired]:
            await c.close()


# ── Run ───────────────────────────────────────────────────────────────────────

FIELDS = ["t_send", "drone_uuid", "model", "batch", "lengths",
          "latency_ms", "status", "error"]


class Stats:
    """Rolling counters for the progress line printed every few seconds."""

    def __init__(self):
        self.reset()
        self.inflight = 0

    def reset(self):
        self.ok = self.err = self.dropped = self.no_endpoint = 0
        self.latencies = []


async def send_one(pool, ep, factory, model, writer, stats):
    inputs, outputs, batch, lengths = factory.make()
    t_send = time.time()
    t0 = time.perf_counter()
    status, error = "ok", ""
    try:
        await pool.client(ep).infer(
            model, inputs, outputs=outputs, client_timeout=REQUEST_TIMEOUT_S
        )
    except Exception as e:
        status, error = "error", str(e).splitlines()[0][:200]
    latency_ms = (time.perf_counter() - t0) * 1000
    if status == "ok":
        stats.ok += 1
        stats.latencies.append(latency_ms)
    else:
        stats.err += 1
    writer.writerow([f"{t_send:.4f}", ep["drone_uuid"], model, batch,
                     json.dumps(lengths, separators=(",", ":")),
                     f"{latency_ms:.2f}", status, error])


async def run_worker(args, cfg, worker, n_workers, run_dir, events, go, start_wall):
    """
    One sender process: the whole profile at 1/n_workers of the rate, into its
    own requests.<worker>.csv. Progress goes to the parent through `events`.
    """
    model = cfg["model"]
    # Same profile seed in every worker, so random walks have the same shape;
    # arrivals and payloads get their own stream per worker.
    profile = Profile(cfg, (args.scale or cfg.get("scale", 1.0)) / n_workers, args.seed)
    rng = np.random.default_rng(args.seed + worker)
    max_inflight = math.ceil(args.max_inflight / n_workers)

    pool = EndpointPool(args.endpoints)
    stats = Stats()
    tasks: set[asyncio.Task] = set()

    async with aiohttp.ClientSession() as session:
        await pool.refresh(session)
        if not pool.ready:
            events.put(("error", worker, f"no ready endpoints in {args.endpoints}"))
            return
        model_config = (await pool.client(pool.ready[0]).get_model_config(model)).config
        factory = PayloadFactory(model_config, cfg, rng)

        csv_file = open(os.path.join(run_dir, f"requests.{worker}.csv"), "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(FIELDS)

        # All workers start on the parent's signal, at the same wall-clock time.
        events.put(("ready", worker, None))
        await asyncio.get_running_loop().run_in_executor(None, go.wait)
        if not start_wall.value:
            csv_file.close()
            return  # released without a start time: the run was aborted

        async def refresher():
            while True:
                await asyncio.sleep(ENDPOINT_REFRESH_S)
                await pool.refresh(session)

        async def reporter():
            tick = 0
            while True:
                await asyncio.sleep(args.report_every)
                tick += 1
                events.put(("stats", worker, {
                    "tick": tick, "ok": stats.ok, "err": stats.err,
                    "dropped": stats.dropped, "no_ep": stats.no_endpoint,
                    "inflight": len(tasks), "endpoints": len(pool.ready),
                    "latencies": stats.latencies,
                }))
                csv_file.flush()
                stats.reset()

        background = [asyncio.create_task(refresher()), asyncio.create_task(reporter())]

        # Open-loop Poisson arrivals. next_t is an absolute schedule so that
        # time spent in the loop body doesn't make the real rate drift low.
        start = time.monotonic() - (time.time() - start_wall.value)
        next_t = 0.0
        try:
            while next_t < profile.duration:
                rate = profile.rate(next_t)
                if rate <= 0:
                    next_t += 0.1
                    await asyncio.sleep(max(0.0, start + next_t - time.monotonic()))
                    continue
                next_t += rng.exponential(1.0 / rate)
                delay = start + next_t - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)

                ep = pool.next()
                if ep is None:
                    stats.no_endpoint += 1
                    writer.writerow([f"{time.time():.4f}", "", model, "", "", "",
                                     "no_endpoint", ""])
                elif len(tasks) >= max_inflight:
                    # Client-side backpressure: recorded, not silently skipped,
                    # because it means the offered load wasn't actually sent.
                    stats.dropped += 1
                    writer.writerow([f"{time.time():.4f}", ep["drone_uuid"], model,
                                     "", "", "", "dropped", ""])
                else:
                    task = asyncio.create_task(
                        send_one(pool, ep, factory, model, writer, stats))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
        except KeyboardInterrupt:
            pass
        finally:
            if tasks:
                await asyncio.wait(tasks, timeout=REQUEST_TIMEOUT_S)
            for b in background:
                b.cancel()
            csv_file.close()
            await pool.close()


def worker_main(*args):
    """Process entry point for run_worker."""
    try:
        asyncio.run(run_worker(*args))
    except KeyboardInterrupt:
        pass


def run(args, cfg):
    """
    Start the worker processes, print their combined progress, then merge
    their CSVs into requests.csv and write meta.json.
    """
    profile = Profile(cfg, args.scale or cfg.get("scale", 1.0), args.seed)
    n = args.workers or max(1, math.ceil(profile.peak() / WORKER_MAX_RATE))

    name = os.path.splitext(os.path.basename(args.profile))[0]
    run_dir = os.path.join(OUTPUT_DIR, f"{name}-{datetime.now():%Y%m%d-%H%M%S}")
    os.makedirs(run_dir, exist_ok=True)

    # spawn, not fork: gRPC does not survive a fork after it was imported.
    ctx = mp.get_context("spawn")
    events, go, start_wall = ctx.Queue(), ctx.Event(), ctx.Value("d", 0.0)
    procs = [ctx.Process(target=worker_main,
                         args=(args, cfg, i, n, run_dir, events, go, start_wall))
             for i in range(n)]
    for p in procs:
        p.start()

    meta = {
        "profile": args.profile, "profile_cfg": cfg, "scale": profile.scale,
        "model": cfg["model"], "seed": args.seed, "max_inflight": args.max_inflight,
        "workers": n, "duration_s": profile.duration,
    }
    error = None
    try:
        ready = 0
        while ready < n and error is None:
            try:
                kind, worker, msg = events.get(timeout=1)
            except queue.Empty:
                dead = [i for i, p in enumerate(procs) if not p.is_alive()]
                if dead:
                    error = f"error: worker {dead[0]} died during setup (see traceback above)"
                continue
            if kind == "error":
                error = f"error (worker {worker}): {msg}"
            ready += kind == "ready"
        if error:
            raise KeyboardInterrupt

        start_wall.value = meta["start"] = time.time()
        go.set()
        print(f"Writing to {run_dir} ({profile.duration} s, scale={profile.scale}, "
              f"workers={n})", flush=True)

        ticks: dict[int, list[dict]] = {}
        while any(p.is_alive() for p in procs):
            try:
                kind, worker, s = events.get(timeout=1)
            except queue.Empty:
                continue
            if kind != "stats":
                continue
            ticks.setdefault(s["tick"], []).append(s)
            if len(ticks[s["tick"]]) < n:
                continue
            parts = ticks.pop(s["tick"])
            lat = sorted(l for p in parts for l in p["latencies"])
            p50 = lat[len(lat) // 2] if lat else float("nan")
            p99 = lat[int(len(lat) * 0.99)] if lat else float("nan")
            t = time.time() - meta["start"]
            total = {k: sum(p[k] for p in parts)
                     for k in ("ok", "err", "dropped", "no_ep", "inflight")}
            print(f"[{t:7.0f}s] target={profile.rate(t):6.1f}/s  "
                  f"ok={total['ok'] / args.report_every:6.1f}/s  err={total['err']}  "
                  f"dropped={total['dropped']}  no_ep={total['no_ep']}  "
                  f"inflight={total['inflight']}  "
                  f"endpoints={max(p['endpoints'] for p in parts)}  "
                  f"p50={p50:.0f}ms p99={p99:.0f}ms", flush=True)
    except KeyboardInterrupt:
        # Ctrl-C reaches the workers too; let them finish their CSVs. Workers
        # still waiting for the start signal are released and end at once.
        go.set()
    finally:
        # A worker only exits once its queued events are read, so keep reading.
        while any(p.is_alive() for p in procs):
            try:
                while True:
                    events.get_nowait()
            except queue.Empty:
                pass
            for p in procs:
                p.join(timeout=0.1)

    if error or "start" not in meta:
        shutil.rmtree(run_dir)
        sys.exit(error or "interrupted before the start")

    meta["end"] = time.time()
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(run_dir, "requests.csv"), "w") as out:
        out.write(",".join(FIELDS) + "\n")
        for i in range(n):
            part = os.path.join(run_dir, f"requests.{i}.csv")
            with open(part) as f:
                f.readline()  # header
                shutil.copyfileobj(f, out)
            os.remove(part)
    print(f"Done → {run_dir}")


def gpu_uuids(metrics_text: str) -> list[str]:
    """
    GPU UUIDs a Triton instance reports, taken from the labels of its
    nv_gpu_utilization lines, e.g.
        nv_gpu_utilization{gpu_uuid="GPU-ab949071-e0aa-..."} 0
    """
    uuids = set()
    for line in metrics_text.splitlines():
        if line.startswith("nv_gpu_utilization{"):
            uuids.add(line.split('gpu_uuid="')[1].split('"')[0])
    return sorted(uuids)


async def sanity(args, model):
    """
    E0: for every drone, report which GPU it uses and send one real inference request. 
    A drone can be READY (health check passes) and still fail every inference.
    """
    pool = EndpointPool(args.endpoints)
    async with aiohttp.ClientSession() as session:
        listed = pool.read_listed()
        await pool.refresh(session)  # health check → pool.ready
        ready = {ep["drone_uuid"] for ep in pool.ready}
        rng = np.random.default_rng(args.seed)
        cfg = {"batch": {"min": 8, "max": 8}, "shapes": {"pf": [50, 50], "sv": [5, 5]}}

        row = "{:<22} {:<24} {:<6} {:>8}  {:<14} {}"
        print(row.format("DRONE", "HOST", "READY", "LATENCY", "GPU", "RESULT"))
        for ep in listed:
            try:
                url = f"http://{ep['host']}:{ep['metrics_port']}/metrics"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as r:
                    text = await r.text()
                gpu = ",".join(u[:12] for u in gpu_uuids(text)) or "none"
            except Exception as e:
                gpu = f"unreachable ({type(e).__name__})"

            is_ready = ep["drone_uuid"] in ready
            latency, result = "-", "skipped (not ready)"
            if is_ready:
                client = pool.client(ep)
                try:
                    model_config = (await client.get_model_config(model)).config
                    inputs, outputs, _, _ = PayloadFactory(model_config, cfg, rng).make()
                    t0 = time.perf_counter()
                    await client.infer(model, inputs, outputs=outputs,
                                       client_timeout=REQUEST_TIMEOUT_S)
                    latency = f"{1000 * (time.perf_counter() - t0):.0f} ms"
                    result = "ok"
                except Exception as e:
                    result = f"FAILED: {str(e).splitlines()[0][:120]}"
            print(row.format(ep["drone_uuid"], ep["host"], "yes" if is_ready else "no",
                             latency, gpu, result))

        await pool.close()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("profile", nargs="?", help="profile YAML (see profiles/)")
    p.add_argument("--endpoints", default=DEFAULT_ENDPOINTS)
    p.add_argument("--scale", type=float, help="override the profile's rate scale")
    p.add_argument("--max-inflight", type=int, default=512,
                   help="in-flight cap over all workers")
    p.add_argument("--workers", type=int,
                   help=f"sender processes (default: peak rate / {WORKER_MAX_RATE})")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report-every", type=float, default=10.0)
    p.add_argument("--sanity", action="store_true", help="E0: one request per endpoint")
    p.add_argument("--model", default="particlenet_AK4_PT",
                   help="model for --sanity (profiles set their own)")
    args = p.parse_args()

    if args.sanity:
        asyncio.run(sanity(args, args.model))
        return
    if not args.profile:
        p.error("a profile is required unless --sanity is given")
    with open(args.profile) as f:
        cfg = yaml.safe_load(f)
    run(args, cfg)


if __name__ == "__main__":
    main()
