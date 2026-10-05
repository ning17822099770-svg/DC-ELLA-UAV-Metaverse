# Reproducing the results

There are two levels:

1. **Rebuild the figures from the saved data** (seconds, no training). Run `python reproduce_figures.py` in
   each experiment folder; see the main [README](../README.md#quick-start-regenerate-the-paper-figures-no-training).
2. **Re-run the simulations** (hours on CPU), described below.

Always run scripts from **inside** the experiment folder. All paths are relative to the working directory.

## 1. Single environment: `experiments/single_env` (Figs. 3–7, 10, 11)

| Step | Command | Writes |
|---|---|---|
| DC-ELLA pretraining (25 ep., `λ^pre`) + test (100 ep., `λ^test`) | `python main_dc_ella_single_env.py` | `pretrained_params/`, `trained_well_params/`, `task_analyse/`, `plot_data_cases_old_transfer/<label>_{pretrain,test}_data.pkl` |
| PG-ELLA baseline | `python baseline_pg_ella.py` | `plot_data_cases/<label>_pg_ella_test_data.pkl` |
| Local learner + transfer baseline (and DC-ELLA test) | `python baseline_transfer_learning.py` | `plot_data_cases_30M_bandwidth/<label>_{test,base_test,transfer_baseline}_data.pkl` |

Choose the caching scheme in `parallel_compare_conditions()`:

```python
cases = [
    {"label": "Only_Hit",       "enable_server_caching": True, "enable_similarity": False, "seed": 3},
    {"label": "Hit+Similarity", "enable_server_caching": True, "enable_similarity": True,  "seed": 3},
]
```

Seed 3 was used for the paper.

The fixed-caching study (Figs. 10–11) is a 150-episode local-learner run. The policy learns under preference
option 1 for 100 episodes; then the caching strategy is frozen and the preferences switch to option 2 (requests
cyclically shifted across servers). Its data is in `plot_data_cases_fixed_caching/`.

## 2. Dynamic environment, similar preferences: `experiments/multi_env_same_preference` (Fig. 8)

`python main_dc_ella_multi_env_same_pref.py`. The 100-episode test phase steps through the 4 configurations in
`TEST_PARAMS_LIST` (same semantic preferences, different task loads) and switches every 25 episodes
(`ENV_SWITCH_EPISODES`). By default `parallel_compare_conditions()` runs only the
transfer baseline (`results_3`). Uncomment `results_1` / `results_2` to also run DC-ELLA / the local learner.

## 3. Dynamic environment, different preferences: `experiments/multi_env_diff_preference` (Fig. 9)

`python main_dc_ella_multi_env_diff_pref.py`: pretraining over 75 episodes, then a 150-episode test phase
in which the environment (content preferences and load) switches every 50 episodes.

## From run outputs to plot inputs

The drivers name their outputs after the run type, while the plotting code expects `<scheme>_<learner>_data.pkl`.
After a run, copy/rename the files into the data folder used by the figure:

| Driver output | Rename to |
|---|---|
| `<label>_test_data.pkl` (DC-ELLA test) | `<label>_DC-ELLA_data.pkl` |
| `<label>_base_test_data.pkl` (`run_mappo_base_test`) | `<label>_local_learner_data.pkl` |
| `<label>_transfer_baseline_data.pkl` | `<label>_transfer_data.pkl` |
| `<label>_pg_ella_test_data.pkl` | `<label>_PG-ELLA_data.pkl` |
| `<label>_pretrain_data.pkl` | unchanged |

`reproduce_figures.py` lists which data folder each figure reads.

## Things to know

* Training overwrites `pretrained_params/`, `trained_well_params/` and the `plot_data_*` files it writes to.
* The test phase loads `pretrained_params/ella_learner.pkl`. Download it first with
  `python tools/fetch_checkpoints.py ...` from the repository root, or run pretraining.
* `device` is `cuda:1` when CUDA is available. See Troubleshooting in the main README.
* `parallel_compare_conditions()` starts one process per case. For debugging, call
  `run_case_with_pretrain_and_test(case)` directly.
* Figs. 6c (service quality vs. number of users) and 11c were produced by earlier runs whose data is not in
  this repository. Only the original images are included (`figures/`).
