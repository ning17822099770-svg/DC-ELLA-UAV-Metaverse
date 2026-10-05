# Dynamic environment with similar preferences: Fig. 8

The user content preferences stay the same while the number of tasks and the arrival rates change. The
100-episode test phase steps through the 4 configurations in `TEST_PARAMS_LIST` and switches environment every
25 episodes (`ENV_SWITCH_EPISODES` in the main script). At every switch, DC-ELLA re-initializes the servers'
policies from the knowledge base `L` and dictionary `D`. It is compared with the PPO local learner and transfer
learning.

| Script | Purpose |
|---|---|
| `main_dc_ella_multi_env_same_pref.py` | Pretraining + dynamic test phase; entry point `parallel_compare_conditions()` |
| `plot_results.py` | Plotting functions (`plot_server_combined_rewards` draws Fig. 8) |
| `reproduce_figures.py` | Rebuilds Fig. 8 from `plot_data_cases/` |

```bash
python reproduce_figures.py   # -> regenerated_figures/fig08_...png
```

Data: `plot_data_cases/Hit+Similarity_{local_learner,transfer,DC-ELLA,pretrain}_data.pkl` (100 test episodes).

Notes:

* `parallel_compare_conditions()` runs only the transfer baseline by default (`results_3`). Uncomment
  `results_1` / `results_2` for DC-ELLA / the local learner.
* `feature_scaler.pkl` and `feature_pca.pkl` are the same files as in `single_env` (they were missing from the
  original folder but are required to build a new DC-ELLA learner).
