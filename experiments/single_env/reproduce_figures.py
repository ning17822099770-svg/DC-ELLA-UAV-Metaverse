"""Regenerate the single-environment figures of the paper from the shipped data.

Usage (from this folder):
    python reproduce_figures.py              # all figures
    python reproduce_figures.py fig04a fig10a

PNGs are written to ./regenerated_figures/<figure>_<original name>.png.
No training is run; the figures are rebuilt from the plot_data_* folders.
Styling may differ slightly from the published versions.
"""
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import plot_results as pr

OUT = Path("regenerated_figures")

# Case labels = <caching scheme>_<learner>, matching <label>_data.pkl in the data folders.
SCHEMES = ["Only_Hit_local_learner", "Only_Hit_DC-ELLA",
           "Hit+Similarity_local_learner", "Hit+Similarity_DC-ELLA"]
LL_BASELINES = ["Hit+Similarity_local_learner", "Hit+Similarity_PG-ELLA",
                "Hit+Similarity_transfer", "Hit+Similarity_DC-ELLA"]
FIXED = ["Only_Hit_local_learner", "Hit+Similarity_local_learner"]
BW30 = "plot_data_cases_30M_bandwidth"
LL_DATA = "plot_data_case_use_now"

# figure id -> (function, args, kwargs). Data folder is fixed inside each
# function unless passed as data_dir.
FIGURES = {
    "fig03":  (pr.plot_server_feature_heatmap, ("Hit+Similarity_local_learner",), {"NUM_UAVS": 5}),
    "fig04a": (pr.plot_server_combined_rewards, (SCHEMES,), {"data_dir": BW30}),
    "fig04b": (pr.plot_critic_server_loss, (SCHEMES,), {}),
    "fig04c": (pr.plot_overall_discount, (SCHEMES,), {"data_dir": BW30}),
    # fig05a/b/c and fig06b come out of one call (several PNGs)
    "fig05_06b": (pr.plot_episode_comparisons_average, (SCHEMES,), {}),
    "fig06a": (pr.plot_saving_breakdown, (SCHEMES, 0, 20), {"data_dir": BW30}),
    "fig07a": (pr.plot_server_combined_rewards, (LL_BASELINES,), {"data_dir": BW30}),
    "fig07b": (pr.plot_overall_discount, (LL_BASELINES,), {"data_dir": BW30}),
    "fig07c": (pr.plot_saving_breakdown, (LL_BASELINES, 0, 20), {"data_dir": LL_DATA}),
    "fig10a": (pr.plot_server_combined_rewards_fixed, (FIXED,), {}),
    "fig10b": (pr.plot_overall_discount_fixed, (FIXED,), {}),
    "fig10c": (pr.plot_hit_similarity_discount, ("Hit+Similarity_local_learner",), {}),
    # last 50 episodes after the preference change; fig11a/b are
    # ..._combined_reward_by_category and ..._computational_saving_by_category.
    # (The published Fig. 11c came from an earlier run and is not reproduced here.)
    "fig11":  (pr.plot_episode_comparisons_average_fixed, (FIXED,), {}),
}


def run(fig_id):
    fn, args, kwargs = FIGURES[fig_id]
    start = time.time()
    fn(*args, **kwargs)
    moved = []
    for png in Path(".").glob("*.png"):
        if png.stat().st_mtime >= start - 1:
            target = OUT / f"{fig_id}_{png.name}"
            shutil.move(str(png), target)
            moved.append(target.name)
    print(f"[{fig_id}] {fn.__name__} -> {', '.join(moved)}")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    wanted = sys.argv[1:] or list(FIGURES)
    unknown = [f for f in wanted if f not in FIGURES]
    if unknown:
        sys.exit(f"Unknown figure id(s): {unknown}. Choose from {list(FIGURES)}")
    for fig_id in wanted:
        run(fig_id)
