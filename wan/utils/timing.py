import os
import time
from contextlib import contextmanager


_GLOBAL_TIMING_STATS: dict[str, dict[str, float]] = {}


def timing_enabled(args=None) -> bool:
    """
    Unified timing switch:
    - Prefer CLI flag: args.print_timing
    - Fallback to env: INFINI_PRINT_TIMING=1/true/yes/on
    """
    if args is not None and bool(getattr(args, "print_timing", False)):
        return True
    v = os.getenv("INFINI_PRINT_TIMING", "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def timing_sync_cuda_enabled(args=None) -> bool:
    """
    When enabled, call torch.cuda.synchronize() before/after timing blocks,
    which gives more accurate GPU timings but can slow execution.
    """
    if args is not None and bool(getattr(args, "print_timing_sync_cuda", False)):
        return True
    v = os.getenv("INFINI_PRINT_TIMING_SYNC_CUDA", "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _is_rank0() -> bool:
    try:
        return int(os.getenv("RANK", "0")) == 0
    except Exception:
        return True


def timing_stats_add(stats: dict, key: str, dt: float) -> None:
    item = stats.get(key)
    if item is None:
        stats[key] = {
            "count": 1,
            "total": float(dt),
            "min": float(dt),
            "max": float(dt),
        }
        return
    item["count"] += 1
    item["total"] += float(dt)
    item["min"] = min(item["min"], float(dt))
    item["max"] = max(item["max"], float(dt))


def merge_timing_stats(dst: dict, src: dict, prefix: str | None = None) -> None:
    """
    Merge src stats into dst stats (summing counts/total, min/min, max/max).
    """
    if not src:
        return
    for k, v in src.items():
        nk = f"{prefix}{k}" if prefix else k
        if nk not in dst:
            dst[nk] = dict(v)
            continue
        d = dst[nk]
        d["count"] = int(d.get("count", 0)) + int(v.get("count", 0))
        d["total"] = float(d.get("total", 0.0)) + float(v.get("total", 0.0))
        d["min"] = min(float(d.get("min", float("inf"))), float(v.get("min", float("inf"))))
        d["max"] = max(float(d.get("max", 0.0)), float(v.get("max", 0.0)))


def get_global_timing_stats_copy() -> dict:
    # Return a shallow copy to avoid accidental external mutation.
    return {k: dict(v) for k, v in _GLOBAL_TIMING_STATS.items()}


def reset_global_timing_stats() -> None:
    _GLOBAL_TIMING_STATS.clear()


def print_timing_summary(stats: dict) -> None:
    if not stats:
        print("[TIMING] summary: (empty)", flush=True)
        return
    print("[TIMING] ===== summary (count/total/avg/min/max) =====", flush=True)
    for k, v in stats.items():
        cnt = int(v["count"])
        total = float(v["total"])
        avg = total / max(cnt, 1)
        mn = float(v["min"])
        mx = float(v["max"])
        print(
            f"[TIMING] {k:60s} | {cnt:6d} | {total:10.3f}s | {avg:9.3f}s | {mn:9.3f}s | {mx:9.3f}s",
            flush=True,
        )
    print("[TIMING] ===== end summary =====", flush=True)


@contextmanager
def timed(
    name: str,
    *,
    enabled: bool | None = None,
    stats: dict | None = None,
    key: str | None = None,
    log: bool = True,
    sync_cuda: bool | None = None,
    rank0_only: bool = True,
):
    """
    Timing context manager.

    - enabled: if None, derived from env INFINI_PRINT_TIMING.
    - stats: if provided, add into this dict; else add into global stats.
    - log: if True, print enter/exit lines; if False, only aggregate stats.
    - sync_cuda: if True, synchronize CUDA before/after for accurate GPU timings.
    - rank0_only: avoid multi-rank log spam; still allows aggregation when enabled.
    """
    if enabled is None:
        enabled = timing_enabled()
    if not enabled:
        yield
        return

    if rank0_only and not _is_rank0():
        yield
        return

    if sync_cuda is None:
        sync_cuda = timing_sync_cuda_enabled()

    if sync_cuda:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    t0 = time.perf_counter()
    if log:
        print(f"[TIMING] >> {name}", flush=True)
    try:
        yield
    finally:
        if sync_cuda:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
        dt = time.perf_counter() - t0
        target = stats if stats is not None else _GLOBAL_TIMING_STATS
        timing_stats_add(target, key or name, dt)
        if log:
            print(f"[TIMING] << {name}: {dt:.3f}s", flush=True)

