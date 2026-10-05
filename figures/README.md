# Figures

The original figures of the paper (IEEE TMC, vol. 25, no. 7, 2026, doi:
[10.1109/TMC.2026.3664868](https://doi.org/10.1109/TMC.2026.3664868)), numbered as in the paper.
The "Reproducible" column says whether `experiments/*/reproduce_figures.py` rebuilds the figure from the
data in this repository.

| File | Paper figure | Reproducible |
|---|---|---|
| `fig01_system_model.png` | Fig. 1: semantic-aware caching enabled by deep lifelong learning | drawing |
| `fig02_dc_ella_design.png` | Fig. 2: Deep-Centralized-ELLA algorithm design | drawing |
| `fig03_request_distribution_heatmap.png` | Fig. 3: distribution of content requests per server | `single_env` `fig03` |
| `fig04a_reward_curve.png` | Fig. 4a: reward, caching schemes × learners | `single_env` `fig04a` |
| `fig04b_critic_loss_curve.png` | Fig. 4b: critic loss | `single_env` `fig04b` |
| `fig04c_saving_curve.png` | Fig. 4c: overall saving | `single_env` `fig04c` |
| `fig05a_reward_by_category.png` | Fig. 5a: reward per server category | `single_env` `fig05_06b` |
| `fig05b_saving_by_category.png` | Fig. 5b: saving per server category | `single_env` `fig05_06b` |
| `fig05c_queue_length_by_category.png` | Fig. 5c: queue length per server category | `single_env` `fig05_06b` |
| `fig06a_saving_breakdown_first20.png` | Fig. 6a: hit / reuse saving, first 20 test episodes | `single_env` `fig06a` |
| `fig06b_time_energy_saving.png` | Fig. 6b: computational time and energy savings | `single_env` `fig05_06b` |
| `fig06c_service_quality.png` | Fig. 6c: service quality ratio vs. number of users | no (earlier user-scaling runs) |
| `fig07a_reward_vs_ll_baselines.png` | Fig. 7a: reward vs. PG-ELLA / transfer learning | `single_env` `fig07a` |
| `fig07b_saving_vs_ll_baselines.png` | Fig. 7b: saving vs. PG-ELLA / transfer learning | `single_env` `fig07b` |
| `fig07c_saving_breakdown_first20_vs_ll_baselines.png` | Fig. 7c: first-20-episode saving vs. baselines | `single_env` `fig07c` |
| `fig08_multi_env_same_preference_reward.png` | Fig. 8: dynamic environment, similar preferences | `multi_env_same_preference` |
| `fig09_multi_env_diff_preference_reward.png` | Fig. 9: dynamic environment, different preferences | `multi_env_diff_preference` |
| `fig10a_reward_fixed_caching.png` | Fig. 10a: reward, fixed caching | `single_env` `fig10a` |
| `fig10b_saving_fixed_caching.png` | Fig. 10b: saving, fixed caching | `single_env` `fig10b` |
| `fig10c_hit_vs_reuse_fixed_caching.png` | Fig. 10c: hit vs. reuse saving | `single_env` `fig10c` |
| `fig11a_last50_reward.png` | Fig. 11a: reward per category, last 50 episodes | `single_env` `fig11` |
| `fig11b_last50_saving.png` | Fig. 11b: saving per category, last 50 episodes | `single_env` `fig11` |
| `fig11c_time_energy_fixed_caching.png` | Fig. 11c: time and energy savings, fixed caching | no (earlier run) |

Regenerated figures contain the same values as the originals; styling (resolution, legend size) may differ.

The figures are reproduced from the paper, © 2026 IEEE.
