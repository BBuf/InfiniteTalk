from __future__ import annotations

import os
import time
from contextlib import contextmanager


@contextmanager
def timed_stage(name: str, enabled: bool = True, stats: dict | None = None, key: str | None = None):
    """
    Print timing using print() (flush=True) to avoid being swallowed by logging.
    Mirrors `generate_infinitetalk.py` behavior.
    """
    if not enabled:
        yield
        return
    t0 = time.perf_counter()
    print(f"[TIMING] >> {name}", flush=True)
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        if stats is not None:
            timing_stats_add(stats, key or name, dt)
        print(f"[TIMING] << {name}: {dt:.3f}s", flush=True)


def timing_enabled(args=None) -> bool:
    if args is not None and getattr(args, "print_timing", False):
        return True
    v = os.getenv("INFINI_PRINT_TIMING", "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def timing_stats_add(stats: dict, key: str, dt: float) -> None:
    item = stats.get(key)
    if item is None:
        stats[key] = {"count": 1, "total": float(dt), "min": float(dt), "max": float(dt)}
        return
    item["count"] += 1
    item["total"] += float(dt)
    item["min"] = min(item["min"], float(dt))
    item["max"] = max(item["max"], float(dt))


def timing_summary_lines(stats: dict) -> list[str]:
    if not stats:
        return ["[TIMING] summary: (empty)"]
    lines = ["[TIMING] ===== summary (count/total/avg/min/max) ====="]
    for k, v in stats.items():
        cnt = int(v["count"])
        total = float(v["total"])
        avg = total / max(cnt, 1)
        mn = float(v["min"])
        mx = float(v["max"])
        lines.append(
            f"[TIMING] {k:40s} | {cnt:4d} | {total:10.3f}s | {avg:9.3f}s | {mn:9.3f}s | {mx:9.3f}s"
        )
    lines.append("[TIMING] ===== end summary =====")
    return lines

