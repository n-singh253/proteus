"""Autonomous-mode quality on synthetic libraries with known structure.

Usage: uv run python benchmarks/quality.py [--seeds 3]

Reports the chosen playlist count and agreement with the generating groups (adjusted Rand
index). Matching the generating count is evidence, not proof, of a good choice: ambiguous
scenarios have several defensible answers.
"""

from __future__ import annotations

import argparse
import time

from sklearn.metrics import adjusted_rand_score
from synthetic import make_library, structureless

from proteus_music.engine.pipeline import plan_playlists
from proteus_music.models.organization import OrganizationConfig, OrganizationMode

SCENARIOS = {
    "clear 4x250": lambda s: make_library([250] * 4, seed=s),
    "clear 8x150": lambda s: make_library([150] * 8, seed=s),
    "imbalanced 600/200/80/40/20": lambda s: make_library([600, 200, 80, 40, 20], seed=s),
    "noisy 5x200 (30% noise)": lambda s: make_library([200] * 5, seed=s, noise=0.3),
    "ambiguous 5x200 (60% noise)": lambda s: make_library([200] * 5, seed=s, noise=0.6),
    "hierarchical 3 families x 3": lambda s: make_library([100] * 9, seed=s, families=3),
    "30% missing 4x200": lambda s: make_library([200] * 4, seed=s, missing=0.3),
    "small 2x15": lambda s: make_library([15, 15], seed=s),
    "tiny single group (12)": lambda s: make_library([12], seed=s),
    "structureless 1000": lambda s: (structureless(1000, seed=s), None),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--algorithm", default="kmeans-merge")
    args = parser.parse_args()
    config = OrganizationConfig(mode=OrganizationMode.AUTO, algorithm=args.algorithm)
    print(f"{'scenario':32} {'groups':>6} {'chosen':>12} {'ARI':>12} {'seconds':>8}")
    for name, build in SCENARIOS.items():
        chosen, agreement, elapsed, groups = [], [], 0.0, "-"
        for seed in range(args.seeds):
            descriptors, truth = build(seed)
            start = time.perf_counter()
            plan = plan_playlists(descriptors, config.model_copy(update={"seed": seed}))
            elapsed += time.perf_counter() - start
            labels = {t: p.index for p in plan.playlists for t in p.track_ids}
            chosen.append(len(plan.playlists))
            if truth is not None:
                groups = str(len(set(truth)))
                predicted = [labels[d.track_id] for d in descriptors]
                agreement.append(adjusted_rand_score(truth, predicted))
        ari = " ".join(f"{a:.2f}" for a in agreement) or "-"
        counts = " ".join(map(str, chosen))
        print(f"{name:32} {groups:>6} {counts:>12} {ari:>12} {elapsed / args.seeds:>8.2f}")


if __name__ == "__main__":
    main()
