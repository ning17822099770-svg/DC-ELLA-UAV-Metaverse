# Dynamic environment with different preferences: Fig. 9

Both the user content preferences (the window of requested Visual Genome contents) and the task load change
between environments. Pretraining runs for 75 episodes; the 150-episode test phase switches environment every
50 episodes (`version = ep // 50`). DC-ELLA is compared with the PPO local learner and transfer learning
(`run_transfer_learning_baseline`).

| Script | Purpose |
|---|---|
| `main_dc_ella_multi_env_diff_pref.py` | Pretraining + dynamic test phase; entry point `parallel_compare_conditions()` |
| `plot_results.py` | Plotting functions (`plot_server_combined_rewards` draws Fig. 9) |
| `reproduce_figures.py` | Rebuilds Fig. 9 from `plot_data_cases/` |

```bash
python reproduce_figures.py   # -> regenerated_figures/fig09_...png
```

Data: `plot_data_cases/Hit+Similarity_{local_learner,transfer,DC-ELLA,pretrain}_data.pkl` (150 test episodes).
`pretrained_params/` and `trained_well_params/` hold the PPO checkpoints (in git) and the DC-ELLA learners
(release assets).
