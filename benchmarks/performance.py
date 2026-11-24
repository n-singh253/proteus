"""Engine runtime and peak memory on synthetic libraries, per organization mode.

Usage: uv run python benchmarks/performance.py [--sizes 1000 10000] [--algorithm NAME]

Each case runs in a fresh subprocess so peak resident memory is per case. Library
generation is excluded from timings. The engine has no feature cache yet, so every run is
cold; there is no warm-cache variant to report.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time

from synthetic import make_library

from proteus_music.engine.pipeline import plan_playlists
from proteus_music.models.organization import Category, OrganizationConfig, OrganizationMode

GROUPS = 20
CATEGORY_NAMES = [f"genre {g}-0" for g in range(5)]
CASES = (
    "fixed",
    "fixed-constrained",
    "range",
    "categories",
    "partial-categories",
    "auto",
    "auto-constrained",
)


def _config(case: str, n: int, algorithm: str) -> OrganizationConfig:
    categories = tuple(Category(name=c) for c in CATEGORY_NAMES)
    cases = {
        "fixed": {"playlists": 15},
        "fixed-constrained": {"playlists": 15, "min_size": n // 30, "max_size": n // 10},
        "range": {"min_playlists": 5, "max_playlists": 25},
        "categories": {"categories": categories},
        "partial-categories": {"playlists": 15, "categories": categories},
        "auto": {"mode": OrganizationMode.AUTO},
        "auto-constrained": {"mode": OrganizationMode.AUTO, "max_size": n // 8},
    }
    return OrganizationConfig(algorithm=algorithm, **cases[case])


def _peak_mb() -> float | None:
    try:
        import resource
    except ImportError:  # Windows.
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def run_case(case: str, n: int, algorithm: str) -> dict:
    sizes = [max(1, round(n * w)) for w in _weights()]
    descriptors, _ = make_library(sizes, seed=0)
    config = _config(case, len(descriptors), algorithm)
    start = time.perf_counter()
    plan = plan_playlists(descriptors, config)
    total = time.perf_counter() - start
    return {
        "total": total,
        **plan.timings,
        "playlists": len(plan.playlists),
        "peak_mb": _peak_mb(),
    }


def _weights() -> list[float]:
    """Moderately imbalanced group sizes (Zipf-like, exponent 0.7)."""
    raw = [1 / (i + 1) ** 0.7 for i in range(GROUPS)]
    return [r / sum(raw) for r in raw]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 50000, 100000])
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=CASES)
    parser.add_argument("--algorithm", default="kmeans-merge")
    parser.add_argument("--child", nargs=2, metavar=("CASE", "SIZE"))
    args = parser.parse_args()
    if args.child:
        print(json.dumps(run_case(args.child[0], int(args.child[1]), args.algorithm)))
        return
    print(f"{platform.processor() or platform.machine()} · Python {platform.python_version()}")
    columns = ("features", "selection", "clustering", "assignment", "evaluation")
    header = " ".join(f"{c:>10}" for c in columns)
    print(f"{'case':20} {'n':>7} {'k':>3} {'total s':>8} {header} {'peak MB':>8}")
    for n in args.sizes:
        for case in args.cases:
            command = [sys.executable, __file__, "--algorithm", args.algorithm]
            command += ["--child", case, str(n)]
            output = subprocess.run(command, capture_output=True, text=True, check=True).stdout
            r = json.loads(output)
            stages = " ".join(f"{r.get(c, 0.0):>10.2f}" for c in columns)
            peak = f"{r['peak_mb']:.0f}" if r["peak_mb"] else "n/a"
            print(f"{case:20} {n:>7} {r['playlists']:>3} {r['total']:>8.2f} {stages} {peak:>8}")


if __name__ == "__main__":
    main()
