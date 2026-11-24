"""Stability of organization under seed changes and library subsampling.

Usage: uv run python benchmarks/stability.py [--algorithms kmeans-merge knn-graph] [--runs 5]

For a fixed synthetic library, reports the mean pairwise adjusted Rand index between runs
that differ only in random seed, and between runs on different random 80% subsamples
(compared on the tracks both contain). 1.0 means identical playlists every time.
"""

from __future__ import annotations

import argparse
import itertools
import random

import graph_clustering
import numpy as np
from sklearn.metrics import adjusted_rand_score
from synthetic import make_library

from proteus_music.engine.pipeline import plan_playlists
from proteus_music.models.organization import OrganizationConfig, OrganizationMode

graph_clustering.register()  # Experimental: knn-graph.

SCENARIOS = {
    "clear 8x150": dict(sizes=[150] * 8),
    "imbalanced 600/200/80/40/20": dict(sizes=[600, 200, 80, 40, 20]),
    "noisy 5x200 (30% noise)": dict(sizes=[200] * 5, noise=0.3),
    "ambiguous 5x200 (60% noise)": dict(sizes=[200] * 5, noise=0.6),
}


def labels(descriptors, config: OrganizationConfig) -> dict[str, int]:
    plan = plan_playlists(descriptors, config)
    return {t: p.index for p in plan.playlists for t in p.track_ids}


def agreement(runs: list[dict[str, int]]) -> float:
    scores = []
    for a, b in itertools.combinations(runs, 2):
        shared = sorted(a.keys() & b.keys())
        scores.append(adjusted_rand_score([a[t] for t in shared], [b[t] for t in shared]))
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithms", nargs="+", default=["kmeans-merge", graph_clustering.NAME])
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    print(f"{'scenario':30} {'algorithm':13} {'mode':6} {'seed ARI':>9} {'subsample ARI':>14}")
    for name, spec in SCENARIOS.items():
        spec = dict(spec)
        descriptors, truth = make_library(spec.pop("sizes"), seed=0, **spec)
        for algorithm, mode in itertools.product(args.algorithms, ("exact", "auto")):
            config = OrganizationConfig(algorithm=algorithm, playlists=len(set(truth)))
            if mode == "auto":
                config = OrganizationConfig(algorithm=algorithm, mode=OrganizationMode.AUTO)
            by_seed = [
                labels(descriptors, config.model_copy(update={"seed": s})) for s in range(args.runs)
            ]
            rng = random.Random(1)
            by_sample = [
                labels(rng.sample(descriptors, int(0.8 * len(descriptors))), config)
                for _ in range(args.runs)
            ]
            print(
                f"{name:30} {algorithm:13} {mode:6} {agreement(by_seed):>9.3f} "
                f"{agreement(by_sample):>14.3f}"
            )


if __name__ == "__main__":
    main()
