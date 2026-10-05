"""Regenerate Fig. 8 (dynamic environment, similar preferences) from the shipped data.

Usage (from this folder):  python reproduce_figures.py
Output: ./regenerated_figures/fig08_multi_same_comparison_server_combined_reward_new.png
"""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import plot_results as pr

CASES = ["Hit+Similarity_local_learner", "Hit+Similarity_transfer", "Hit+Similarity_DC-ELLA"]

if __name__ == "__main__":
    out = Path("regenerated_figures")
    out.mkdir(exist_ok=True)
    pr.plot_server_combined_rewards(CASES)  # reads plot_data_cases/
    name = "multi_same_comparison_server_combined_reward_new.png"
    shutil.move(name, out / f"fig08_{name}")
    print(f"[fig08] -> {out / f'fig08_{name}'}")
