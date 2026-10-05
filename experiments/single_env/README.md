# Single (static) semantic environments: Figs. 3–7, 10, 11

Five UAV servers serve users from five semantic environments (Entertainment, Home, Wild, Traffic, Work).
This folder compares:

* **caching schemes:** Only-Hit vs. Hit+Similarity (semantic reuse);
* **learners:** PPO local learner vs. DC-ELLA, plus the PG-ELLA and transfer-learning baselines;
* **fixed caching** after an abrupt preference change (Figs. 10–11).

## Scripts

| Script | Purpose |
|---|---|
| `main_dc_ella_single_env.py` | DC-ELLA: pretraining (25 episodes) + test (100 episodes) |
| `baseline_pg_ella.py` | PG-ELLA baseline (no dictionary `D`) |
| `baseline_transfer_learning.py` | DC-ELLA test + local learner + transfer-learning baseline |
| `plot_results.py` | Plotting functions |
| `reproduce_figures.py` | Rebuild the paper figures from the saved data |
| `fit_feature_scaler_pca.py` | Offline fit of `feature_scaler.pkl` / `feature_pca.pkl` (provenance only) |

## Figures

```bash
python reproduce_figures.py            # everything -> regenerated_figures/
python reproduce_figures.py fig04a     # a single figure
```

| Figure id | Function | Data folder | Cases |
|---|---|---|---|
| `fig03` | `plot_server_feature_heatmap` | `plot_data_cases_30M_bandwidth` | Hit+Similarity_local_learner |
| `fig04a`, `fig04c` | `plot_server_combined_rewards`, `plot_overall_discount` | `plot_data_cases_30M_bandwidth` | {Only_Hit, Hit+Similarity} × {local_learner, DC-ELLA} |
| `fig04b` | `plot_critic_server_loss` | `plot_data_cases_30M_bandwidth` | same as Fig. 4 |
| `fig05_06b` | `plot_episode_comparisons_average` | `plot_data_cases_30M_bandwidth` | same as Fig. 4; produces Fig. 5a (`…combined_reward_by_category`), 5b (`…computational_saving_by_category`), 5c (`…queue_length`) and 6b (`dual_axis_computational_saving`) |
| `fig06a` | `plot_saving_breakdown` (episodes 0–20) | `plot_data_cases_30M_bandwidth` | same as Fig. 4 |
| `fig07a`, `fig07b` | `plot_server_combined_rewards`, `plot_overall_discount` | `plot_data_cases_30M_bandwidth` | Hit+Similarity × {local_learner, PG-ELLA, transfer, DC-ELLA} |
| `fig07c` | `plot_saving_breakdown` (episodes 0–20) | `plot_data_case_use_now` | same as Fig. 7a |
| `fig10a`, `fig10b` | `plot_server_combined_rewards_fixed`, `plot_overall_discount_fixed` | `plot_data_cases_fixed_caching` | {Only_Hit, Hit+Similarity}_local_learner |
| `fig10c` | `plot_hit_similarity_discount` | `plot_data_cases_fixed_caching` | Hit+Similarity_local_learner |
| `fig11` | `plot_episode_comparisons_average_fixed` (last 50 episodes) | `plot_data_cases_fixed_caching` | same as Fig. 10a; produces Fig. 11a (`…combined_reward_by_category`) and 11b (`…computational_saving_by_category`) |

All figures above were checked against the original paper figures (`../../figures/`). Note that Fig. 7c
uses `plot_data_case_use_now` while Figs. 7a/7b use `plot_data_cases_30M_bandwidth` (whose PG-ELLA run has
50 episodes). The published Figs. 6c and 11c came from earlier runs whose data is not part of this repository.

## Result folders

| Folder | Content |
|---|---|
| `plot_data_cases_30M_bandwidth/` | Main comparison run (B = 30 MHz, Table II): all schemes × learners (Figs. 3–6, 7a–b) |
| `plot_data_case_use_now/` | Run used for the first-20-episode baseline comparison (Fig. 7c); its `pretrained_params/` holds the matching PPO checkpoints |
| `plot_data_cases_fixed_caching/` | 150-episode fixed-caching runs, preference change at episode 100 (Figs. 10–11) |
| `pretrained_params/`, `trained_well_params/` | Pretrained PPO networks (in git) and DC-ELLA learners (release assets) |
| `task_analyse/` | Task-detection logs of pretraining and test |

See [../../docs/reproducing_results.md](../../docs/reproducing_results.md) for re-running the simulations.
