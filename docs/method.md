# Method summary and paper-to-code map

This page summarizes the paper and points to the code that implements each part.
Equation numbers follow the paper.

## 1. System model (Sec. II)

* **Semantic representation.** RelTR extracts `<subject, predicate, object>` triples from images/frames.
  Subject/object labels are embedded with MPNet (`sentence-transformers/all-mpnet-base-v2`, mean pooling,
  L2 normalization). Their cosine similarities are precomputed in `subject_similarity_matrix.mat` and
  `object_similarity_matrix.mat`.
* **Communication.** Air-to-ground channel with LoS/NLoS path loss and a Shannon-rate uplink/downlink (Eq. 2).
  Code: `MultiUserEnv._calculate_upload_rate`, `_calculate_downlink_rate`.
* **Rendering.** `T^Ren = w / (Z_j f_j)`, `E^Ren = P^Ren T^Ren` (Eq. 3).

## 2. Semantic-aware caching and rendering (Sec. III)

| Concept | Equation | Code (`Environments_new_feature.py`) |
|---|---|---|
| Content similarity `S^c` (MPNet cosine) | Eq. 4–6 | precomputed `.mat` matrices |
| Environment feature vector and similarity `S^e` | Eq. 9–10 | `compute_uav_metrics_and_similarity` |
| Reuse probability `P^re = α1 S^c + (1-α1) S^e` (α1 = 0.5) | Eq. 11 | `_server_update_queues_with_partial` |
| Average weighted reuse probability (state) | Eq. 12 | `_get_server_obs` |
| Computational saving `S^ave` (hit + reuse) | Eq. 13–14 | `_server_update_queues_with_partial`, `_compute_server_stats_tasks`, `_compute_server_local_reward` |
| Task completion quality `Sa_j(t)` (latency / resolution satisfaction) | Eq. 15 | `step` |
| Server utility `U_j(t) = α3 S^ave + (1-α3) Sa` (reward, `alpha`/`beta` = 0.5) | Eq. 16 | `step` |

`enable_server_caching` / `enable_similarity` select **No-Cache**, **Only-Hit** (hit only) or
**Hit+Similarity** (hit + semantic reuse).

## 3. Optimization problem and MDP (Sec. IV–V-A)

Problem **P1** jointly optimizes user–server association, subject/object caching and reuse decisions to
maximize `Σ_j U_j(t)` (Eq. 17). It is NP-hard and is recast as an MDP per UAV:

* **State:** subject/object request popularity and average weighted reuse probability (200-dim, normalized).
* **Action:** 115 dims = 15 user–server associations + 50 subject + 50 object cache decisions; a
  per-block Top-10 rule enforces the cache capacity `N_F = 10`. Reuse decisions follow deterministically from the state.
* **Reward:** `U_j(t)`.
* **Task feature `Φ`:** policy-independent statistics of recent requests (202-dim), used to detect
  semantic environments (lifelong tasks).

Code: `Server_agent.ServerMAPPOAgent`, `network_define.ServerActorNetNoGRUForELLA`.

## 4. DC-ELLA (Sec. V-C/V-D, Algorithms 1–2)

Policy parameters and environment features share one sparse code per task:

```
θ_m ≈ L s_m        (L ∈ R^{d×k}: shared knowledge base)
Φ_m ≈ D s_m        (D ∈ R^{f×k}: environment-feature dictionary)

min_{L,D,{s_m}}  (1/M) Σ_m [ ||θ̂_m − L s_m||²_{H_m} + λ ||Φ_m − D s_m||² + μ1 ||s_m||_1 ] + μ2 (||L||²_F + ||D||²_F)   (Eq. 23)
```

| Step | Equation | Code (`Server_LL_Learner_with_dictionary.py`) |
|---|---|---|
| Stack `β_m = [θ_m; Φ_m]`, block curvature `Γ_m` | Eq. 24–25 | `learn_multitask_server_actors` |
| Curvature-weighted Lasso for `s_m` | Eq. 26 | `solve_for_s_extended` (CVXPY) |
| Sparse code of a new environment from `D` | Eq. 27 | `compute_s_from_feature` |
| Remove / add task contribution to `A`, `b` | Eq. 28–31 | `remove_old_contribution`, `add_new_contribution` |
| Closed-form update of `L` | Eq. 32 | `update_L_closed_form` |
| Update of `D` | Eq. 33–34 | `update_dictionary_D_closed_form`, `update_dictionary_D` |
| Lifelong task detection (cosine on PCA features, threshold 0.85) | – | `determine_task_id`, `preprocess_feature` |
| Policy push-back `θ̃ = L s_m` | – | `get_reconstructed_theta`, `network_define.load_policy_parameters` |

The PPO local learner supplies `θ̂_m` (the 32×32 `policy_mlp` layer, `extract_policy_parameters`) and a
curvature estimate `H_m` (empirical Fisher / Gauss–Newton / diagonal, `Server_agent._compute_actor_*`).

**Per-episode loop (Algorithm 1)** in the main scripts: (1) servers act with `θ̃ = L s_m` and update PPO
locally; (2) the central learner aggregates features into `Φ`, detects the task, and updates `s_m`, `L`, `D`;
(3) newly detected tasks receive the regenerated `θ̃`.

**Pretraining → test (Algorithm 2):** (i) fast adaptation, where `s` is initialized from the new environment's
features through `D`; (ii) full local PPO updates; (iii) DC-ELLA guidance that refines `s`, `L` and `D`.
This is `run_mappo_pretrain()` followed by `run_mappo_test()`.

**Baselines:**

* *local learner*: PPO only, `run_mappo_base_test()`.
* *PG-ELLA*: no dictionary `D`, `Server_LL_Learner_no_dictionary.py` with `baseline_pg_ella.py`.
* *transfer learning*: fine-tunes the pretrained PPO, `run_transfer_learning_baseline()`.

## 5. Simulation parameters (Table II)

| Parameter | Value |
|---|---|
| Users `I` / UAVs `J` | 15 / 5 |
| UAV height, map size | 50 m, 1000 m × 1000 m |
| Transmit power user / UAV | 0.1 W / 0.5 W |
| Bandwidth user / UAV | 30 MHz / 30 MHz |
| Carrier frequency, noise power | 2.4 GHz, 10^-20.4 W |
| Cache size `N_F` | 10 |
| Episodes `N_e`, steps per episode `S_e` | 100, 30 |
| Latency requirements `T^req` | [2, 2.5, 3, 1.5, 1] s |
| Resolution requirements `s^req` | [2k, 3k, 4k, 0.72k, 1k] |
| CPU frequencies `f_j` | [1, 1.5, 2, 2.5, 3] GHz |
| Computation capabilities `ca_j` (server categories 1–5) | [67, 45, 56, 31, 22] (values used by the code; Table II of the paper prints [67, 56, 40, 31, 21]) |
| Rendering power `P^Ren`, data per cycle `Z_j` | 1.7 W, 0.25 |
| Pretraining / test arrival rates `λ^pre` / `λ^test` | [100, 120, 80, 60, 40] / [200, 120, 100, 30, 20] |
| Semantic environments `M` | 5 (static), 10 (dynamic) |
| PPO | lr 1e-4, γ 0.99, GAE λ 0.95, clip 0.4, entropy 0.1, grad-norm 0.5 |
| DC-ELLA | task-feature dim 202, latent dim 8, μ1 = μ2 = 1e-5, L learning rate 1e-3 |
