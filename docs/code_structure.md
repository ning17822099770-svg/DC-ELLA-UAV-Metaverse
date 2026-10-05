# Code structure

All three experiment folders share the same module layout. Each folder keeps its own copy of the modules
(self-contained design); `Server_agent.py`, `User_agent.py` and `Server_LL_Learner_with_dictionary.py` are
identical across folders, while `Environments_new_feature.py` and `network_define.py` differ slightly per experiment.

```
              main_*.py / baseline_*.py   (pretraining -> test loop, Algorithms 1-2)
                     │
      ┌──────────────┼───────────────────────────────┐
      ▼              ▼                                ▼
Environments_   Server_agent.py               Server_LL_Learner_with_dictionary.py   (DC-ELLA)
new_feature.py  (PPO local learners)          Server_LL_Learner_no_dictionary.py     (PG-ELLA)
(MultiUserEnv)       │
                     ▼
               network_define.py (actor / critic, θ extraction)

plot_results.py      -> figure functions          reproduce_figures.py -> figure id → function → data
```

## Modules

| File | Role |
|---|---|
| `Environments_new_feature.py` | `MultiUserEnv`: users, UAVs, Poisson request generation, A2G rates, hit / reuse / full computation, savings, task quality, reward, server observations and task features. |
| `Server_agent.py` | `ServerMAPPOAgent`: one PPO actor-critic per UAV; action decoding (association + Top-10 caching); curvature estimates `H_m` for DC-ELLA. |
| `network_define.py` | `ServerActorNetNoGRUForELLA` (paper actor, θ = `policy_mlp` weights), `server_GRUCriticNet`, user-side networks. |
| `Server_LL_Learner_with_dictionary.py` | `ServerLifelongLearner`: DC-ELLA (knowledge base `L`, dictionary `D`, sparse codes `s_m`, task detection). |
| `Server_LL_Learner_no_dictionary.py` | PG-ELLA baseline (single_env only). |
| `User_agent.py` | User-side MAPPO agent, imported but not used in the reported experiments. |
| `main_*.py` | Experiment drivers: `run_mappo_pretrain`, `run_mappo_test` (DC-ELLA), `run_mappo_base_test` (local learner), `run_transfer_learning_baseline` (where present), `parallel_compare_conditions` (entry point). |
| `baseline_pg_ella.py`, `baseline_transfer_learning.py` | Variants of the single-env driver for the Fig. 7 baselines. |
| `plot_results.py` | Plotting functions (one PNG per call, reading `plot_data_*`). |
| `reproduce_figures.py` | Maps paper figure ids to plotting functions, case lists and data folders. |
| `fit_feature_scaler_pca.py` | Offline fit of `feature_scaler.pkl` / `feature_pca.pkl` (provenance). |

> The module names (`Environments_new_feature`, `Server_LL_Learner_with_dictionary`, …) are kept from the
> original code on purpose: the pickled learners reference them by name.

## Files inside an experiment folder

| Path | Content | In git? |
|---|---|---|
| `image_label_mapping.pkl`, `*_similarity_matrix.mat`, `global_*_map.pkl` | Semantic input data | yes |
| `feature_scaler.pkl`, `feature_pca.pkl` | Fitted scaler / PCA for task detection | yes |
| `pretrained_params/server_actor_*_non_policy_params.pt`, `server_critic_*_params.pt` | Pretrained PPO networks | yes |
| `pretrained_params/ella_learner.pkl` | Pretrained DC-ELLA learner (`L`, `D`, `s`) | release asset |
| `trained_well_params/ella_learner_final.pkl` | DC-ELLA learner after the test phase | release asset |
| `task_analyse/{pretrain,test}_stats.pkl` | Task-detection logs (task ids per server/episode, policy-change events) | yes |
| `plot_data_*/<case>_data.pkl` | Per-episode statistics used for the figures | yes |

`<case>` = `<scheme>_<learner>`, with scheme ∈ {`Only_Hit`, `Hit+Similarity`} and learner ∈ {`local_learner`,
`DC-ELLA`, `PG-ELLA`, `transfer`, `pretrain`}.

Each `<case>_data.pkl` is a dict:

* `final_stats`: per-episode lists such as `server_combined_reward`, `server_global_reward`,
  `server_critic_loss`, `overall_discount`, `server_hit_rate`, ...
* `final_metrics`: summary scalars.
* `extra_stats`: per server / category / user-environment series (`server_hit_by_uav`, `server_sim_by_uav`,
  `server_queue_by_cat`, `server_combined_by_cat`, `user_local_by_env`, `server_feature_history`, ...).
