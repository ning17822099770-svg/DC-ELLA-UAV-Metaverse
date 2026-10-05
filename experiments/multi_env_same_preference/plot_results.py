"""Plotting utilities for the same-preference dynamic experiment.

plot_server_combined_rewards() produces Fig. 8
(multi_same_comparison_server_combined_reward_new.png) from plot_data_cases/.

Paper: "Deep Lifelong Learning for Adaptive Semantic-Aware Content Reuse in
UAV-Assisted Metaverse", IEEE Transactions on Mobile Computing.
Equation / section numbers refer to the paper.
"""

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

def smooth_curve(vals, window_size=15):
    smoothed = []
    length = len(vals)
    for i in range(length):
        left = max(0, i - window_size + 1)
        window_data = vals[left:i + 1]
        smoothed.append(np.mean(window_data))
    return smoothed

def plot_user_local_rewards(case_labels):
    colors = ['#6CA6CD', '#9ACD32', '#FA8072', '#FFB347', '#FFD966']

    idx = 0
    plt.figure(figsize=(12, 6))
    for case_label in case_labels:
        load_path = f"1_2_plot_data_cases/{case_label}_data.pkl"
        with open(load_path, "rb") as f:
            data_loaded = pickle.load(f)
        final_stats = data_loaded["final_stats"]
        num_episodes = len(final_stats["user_local_reward"])
        x_ep = np.arange(num_episodes) + 1
        plt.plot(x_ep, smooth_curve(final_stats["user_local_reward"]), linestyle='--', marker='o', markevery=5, label=f"{case_label} Local", color=colors[idx % len(colors)])

        idx += 1

    plt.xlabel("Episode")
    plt.ylabel("User Reward")
    plt.title("User Local Rewards Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_user_local_rewards_curve.png")
    plt.close()

def plot_user_global_rewards(case_labels):
    colors = ['#6CA6CD', '#9ACD32', '#FA8072', '#FFB347', '#FFD966']

    plt.figure(figsize=(12, 6))
    idx = 0
    for case_label in case_labels:
        load_path = f"1_2_plot_data_cases/{case_label}_data.pkl"
        with open(load_path, "rb") as f:
            data_loaded = pickle.load(f)
        final_stats = data_loaded["final_stats"]
        num_episodes = len(final_stats["user_global_reward"])
        x_ep = np.arange(num_episodes) + 1
        plt.plot(x_ep, smooth_curve(final_stats["user_global_reward"]), linestyle='-', marker='o', markevery=5, label=f"{case_label} Global", color=colors[idx % len(colors)])

        idx += 1

    plt.xlabel("Episode")
    plt.ylabel("User Reward")
    plt.title("User Global Rewards Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_user_global_rewards_curve.png")
    plt.close()

def plot_server_global_rewards(case_labels):
    colors = ['#6CA6CD', '#9ACD32', '#FA8072', '#FFB347', '#FFD966']
    plt.figure(figsize=(12, 6))
    idx = 0
    for case_label in case_labels:
        load_path = f"1_2_plot_data_cases/{case_label}_data.pkl"
        with open(load_path, "rb") as f:
            data_loaded = pickle.load(f)
        final_stats = data_loaded["final_stats"]
        num_episodes = len(final_stats["server_global_reward"])
        x_ep = np.arange(num_episodes) + 1
        plt.plot(x_ep, smooth_curve(final_stats["server_global_reward"]), linestyle='-', marker='o', markevery=5, label=f"{case_label} Global", color=colors[idx % len(colors)])

        idx += 1
    plt.xlabel("Episode")
    plt.ylabel("Server Global Reward")
    plt.title("Server Global Rewards Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_server_global_rewards_curve.png")
    plt.close()


def plot_server_combined_rewards(case_labels):
    # colors = ['#6CA6CD', '#9ACD32', '#FA8072', '#FFB347', '#FFD966'] , color=colors[idx % len(colors)]
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']

    plt.figure(figsize=(10, 6))
    idx = 0
    max_episodes = 0  # largest episode count, used to place the vertical lines

    for case_label in case_labels:
        load_path = f"plot_data_cases/{case_label}_data.pkl"
        with open(load_path, "rb") as f:
            data_loaded = pickle.load(f)
        final_stats = data_loaded["final_stats"]

        # Load the data and update the max episode count
        if case_label == "Hit+Similarity_LL":
            y_data = final_stats["server_global_reward"]
        else:
            y_data = final_stats["server_combined_reward"]

        num_episodes = len(y_data)
        if num_episodes > max_episodes:
            max_episodes = num_episodes

        x_ep = np.arange(num_episodes) + 1

        # Keep the original plotting style
        plt.plot(x_ep, smooth_curve(y_data), linestyle='-', marker='o', markevery=1,
                 label=f"{case_label}", color=colors[idx % len(colors)])
        idx += 1

    # Draw a dashed vertical line every 25 episodes (environment switches)
    # Black/gray dashed style so the lines are not confused with the data
    for x_line in range(25, max_episodes + 1, 25):
        plt.axvline(x=x_line, linestyle='--', color='black', linewidth=2, alpha=0.5)

    plt.xlabel("Episode", fontsize=18)
    plt.ylabel("Server Global Combined Reward", fontsize=18)
    # plt.title("Server Global Combined Rewards Comparison", fontsize=14)
    plt.legend(fontsize=18)
    plt.grid(True)  # keep the original grid style
    plt.savefig("multi_same_comparison_server_combined_reward_new.png")
    plt.close()

    # all_hit = np.mean([extra_stats["server_hit_by_uav"][uav] for uav in extra_stats["server_hit_by_uav"]], axis=0)
    # all_sim = np.mean([extra_stats["server_sim_by_uav"][uav] for uav in extra_stats["server_sim_by_uav"]], axis=0)

def plot_overall_discount(case_labels):
    # colors = ['#FA8072', '#9ACD32', '#6CA6CD' , '#FFB347', '#FFD966'] , color=colors[idx % len(colors)]

    plt.figure(figsize=(12, 6))
    idx = 0
    for case_label in case_labels:
        load_path = f"plot_data_cases/{case_label}_data.pkl"
        with open(load_path, "rb") as f:
            data_loaded = pickle.load(f)
        final_stats = data_loaded["final_stats"]
        extra_stats = data_loaded["extra_stats"]
        num_episodes = len(final_stats["overall_discount"])

        all_hit = np.mean([extra_stats["server_hit_by_uav"][uav] for uav in extra_stats["server_hit_by_uav"]], axis=0)
        all_sim = np.mean([extra_stats["server_sim_by_uav"][uav] for uav in extra_stats["server_sim_by_uav"]], axis=0)

        x_ep = np.arange(num_episodes) + 1
        plt.plot(x_ep, smooth_curve(all_hit + all_sim), linestyle='-', marker='o', markevery=1, label=f"{case_label}")

        idx+=1

    plt.xlabel("Episode",fontsize=14)
    plt.ylabel("Overall Saving",fontsize=14)
    # plt.title("Overall Saving Comparison",fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig("comparison_overall_saving_curve.png")
    plt.close()


def plot_critic_user_loss(case_labels):
    colors = ['#6CA6CD', '#9ACD32', '#FA8072', '#FFB347', '#FFD966']

    plt.figure(figsize=(12, 6))

    idx = 0
    for case_label in case_labels:
        with open(f"1_2_plot_data_cases/{case_label}_data.pkl", "rb") as f:
            data_loaded = pickle.load(f)

        user_critic_loss = data_loaded["final_stats"]["user_critic_loss"]
        server_critic_loss = data_loaded["final_stats"]["server_critic_loss"]
        episodes = np.arange(1, len(user_critic_loss) + 1)

        plt.plot(episodes, user_critic_loss, marker='o', markevery=5, label=f"User Critic Loss ({case_label})", color=colors[idx % len(colors)])
        # plt.plot(episodes, server_critic_loss, marker='x', linestyle='--', label=f"Server Critic Loss ({case_label})")

        idx += 1

    plt.xlabel("Episode")
    plt.ylabel("Critic Loss")
    plt.title("Critic Loss Comparison Over Episodes")
    plt.grid(True)
    plt.legend()
    plt.savefig("comparison_critic_user_loss_curve.png")
    plt.close()

def plot_critic_server_loss(case_labels):
    # colors = ['#6CA6CD', '#9ACD32', '#FA8072', '#FFB347', '#FFD966'] , color=colors[idx % len(colors)]

    plt.figure(figsize=(12, 6))

    idx = 0
    for case_label in case_labels:
        with open(f"plot_data_cases/{case_label}_data.pkl", "rb") as f:
            data_loaded = pickle.load(f)

        user_critic_loss = data_loaded["final_stats"]["user_critic_loss"]
        server_critic_loss = data_loaded["final_stats"]["server_critic_loss"]
        episodes = np.arange(1, len(user_critic_loss) + 1)

        # plt.plot(episodes, user_critic_loss, marker='o', label=f"User Critic Loss ({case_label})")
        plt.plot(episodes, server_critic_loss, marker='o', markevery=1, label=f"Server Critic Loss ({case_label})")

        idx += 1

    plt.xlabel("Episode",fontsize=14)
    plt.ylabel("Critic Loss",fontsize=14)
    # plt.title("Critic Loss Comparison Over Episodes", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.savefig("comparison_critic_server_loss_curve.png")
    plt.close()

def plot_final_episode_comparisons(case_labels):
    categories = ['1', '2', '3', '4', '5']
    x = np.arange(len(categories))
    width = 0.25

    # Server Queue Length
    plt.figure(figsize=(10, 6))
    for idx, case_label in enumerate(case_labels):
        load_path = f"1_2_plot_data_cases/{case_label}_data.pkl"
        with open(load_path, "rb") as f:
            data_loaded = pickle.load(f)
        extra_stats = data_loaded["extra_stats"]
        queue_means = [extra_stats["server_queue_by_cat"].get(int(cat), [0.0])[-1] for cat in categories]
        plt.bar(x + idx * width - width, queue_means, width, label=f"{case_label}")

    plt.xticks(x, categories)
    plt.xlabel("Server Category")
    plt.ylabel("Queue Length")
    plt.title("Server Queue Length by Category")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_server_queue_length.png")
    plt.close()

    # Server Combined Reward
    plt.figure(figsize=(10, 6))
    for idx, case_label in enumerate(case_labels):
        load_path = f"1_2_plot_data_cases/{case_label}_data.pkl"
        with open(load_path, "rb") as f:
            data_loaded = pickle.load(f)
        extra_stats = data_loaded["extra_stats"]
        reward_means = [extra_stats["server_combined_by_cat"].get(int(cat), [0.0])[-1] for cat in categories]
        plt.bar(x + idx * width - width, reward_means, width, label=f"{case_label}")

    plt.xticks(x, categories)
    plt.xlabel("Server Category")
    plt.ylabel("Combined Reward")
    plt.title("Server Combined Reward by Category")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_server_combined_reward_by_category.png")
    plt.close()

    # Server Hit Rate by Category
    plt.figure(figsize=(10, 6))
    for idx, case_label in enumerate(case_labels):
        load_path = f"1_2_plot_data_cases/{case_label}_data.pkl"
        with open(load_path, "rb") as f:
            data_loaded = pickle.load(f)
        extra_stats = data_loaded["extra_stats"]
        hit_rate_means = [extra_stats["server_hit_by_cat"].get(int(cat), [0.0])[-1] for cat in categories]
        plt.bar(x + idx * width - width, hit_rate_means, width, label=f"{case_label}")

    plt.xticks(x, categories)
    plt.xlabel("Server Category")
    plt.ylabel("Hit Rate")
    plt.title("Server Hit Rate by Category")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_server_hit_rate_by_category.png")
    plt.close()

    # User Local and Combined Reward
    user_types = ['Entertainment', 'Home', 'Wild', 'Traffic', 'Work']
    x_types = np.arange(len(user_types))
    plt.figure(figsize=(12, 6))
    for idx, case_label in enumerate(case_labels):
        load_path = f"1_2_plot_data_cases/{case_label}_data.pkl"
        with open(load_path, "rb") as f:
            data_loaded = pickle.load(f)
        extra_stats = data_loaded["extra_stats"]
        local_means = [extra_stats["user_local_by_env"].get(env, [0.0])[-1] for env in user_types]
        # global_means = [extra_stats["user_combined_by_env"].get(env, [0.0])[-1] for env in user_types]
        plt.bar(x_types + idx * width - width, local_means, width / 2, label=f"{case_label} Local")
        # plt.bar(x_types + idx * width - width / 2, global_means, width / 2, label=f"{case_label} Combined")

    plt.xticks(x_types, ['1', '2', '3', '4', '5'])
    plt.xlabel("User Environment")
    plt.ylabel("Reward")
    plt.title("User Local  Reward by Environment")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_user_rewards_by_env.png")
    plt.close()

    # Server service quality
    plt.figure(figsize=(10, 6))
    for idx, case_label in enumerate(case_labels):
        load_path = f"1_2_plot_data_cases/{case_label}_data.pkl"
        with open(load_path, "rb") as f:
            data_loaded = pickle.load(f)
        extra_stats = data_loaded["extra_stats"]
        service_quality = np.array([
            (extra_stats["server_combined_by_cat"].get(int(cat), [0.0])[-1] -
            extra_stats["server_local_by_cat"].get(int(cat), [0.0])[-1] * 0.5) / 0.5
            for cat in categories
        ])
        plt.bar(x + idx * width - width, service_quality, width, label=f"{case_label}")

    plt.xticks(x, categories)
    plt.xlabel("Server Category")
    plt.ylabel("service_quality")
    plt.title("Server service_quality by Category")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_server_service_quality_by_category.png")
    plt.close()

    categories = ['0', '1', '2', '3', '4']
    # Server computational_saving
    plt.figure(figsize=(10, 6))
    for idx, case_label in enumerate(case_labels):
        load_path = f"1_2_plot_data_cases/{case_label}_data.pkl"
        with open(load_path, "rb") as f:
            data_loaded = pickle.load(f)
        extra_stats = data_loaded["extra_stats"]
        computational_saving = np.array([
            extra_stats["server_hit_by_uav"].get(int(cat), [0.0])[-1] +
            extra_stats["server_sim_by_uav"].get(int(cat), [0.0])[-1]
            for cat in categories
        ])
        plt.bar(x + idx * width - width, computational_saving, width, label=f"{case_label}")

    plt.xticks(x, categories)
    plt.xlabel("Server Category")
    plt.ylabel("computational_saving")
    plt.title("Server computational_saving by Category")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_server_computational_saving_by_category.png")
    plt.close()
#
# def plot_final_episode_comparisons_average(case_labels):
#     categories = ['1', '2', '3', '4', '5']
#     x = np.arange(len(categories))
#     width = 0.25
#     colors = ['#6CA6CD', '#9ACD32', '#FA8072', '#FFB347', '#FFD966']
#     idx = 0
#
#     # Server Queue Length (Mean)
#     plt.figure(figsize=(10, 6))
#     for idx, case_label in enumerate(case_labels):
#         with open(f"plot_data_cases/{case_label}_data.pkl", "rb") as f:
#             data_loaded = pickle.load(f)
#         extra_stats = data_loaded["extra_stats"]
#         queue_means = [np.mean(extra_stats["server_queue_by_cat"].get(int(cat), [0.0])) for cat in categories]
#         plt.bar(x + idx * width - width, queue_means, width, label=f"{case_label}")
#
#     plt.xticks(x, categories)
#     plt.xlabel("Server Category")
#     plt.ylabel("Average Queue Length")
#     plt.title("Average Server Queue Length by Category")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig("comparison_average_server_queue_length.png")
#     plt.close()
#
#     # Server Combined Reward (Mean)
#     plt.figure(figsize=(10, 6))
#     for idx, case_label in enumerate(case_labels):
#         with open(f"plot_data_cases/{case_label}_data.pkl", "rb") as f:
#             data_loaded = pickle.load(f)
#         extra_stats = data_loaded["extra_stats"]
#         reward_means = [np.mean(extra_stats["server_combined_by_cat"].get(int(cat), [0.0])) for cat in categories]
#         plt.bar(x + idx * width - width, reward_means, width, label=f"{case_label}")
#
#     plt.xticks(x, categories)
#     plt.xlabel("Server Category")
#     plt.ylabel("Average Combined Reward")
#     plt.title("Average Server Combined Reward by Category")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig("comparison_average_server_combined_reward_by_category.png")
#     plt.close()
#
#     # Server Hit Times by Category (Mean)
#     plt.figure(figsize=(10, 6))
#     for idx, case_label in enumerate(case_labels):
#         with open(f"plot_data_cases/{case_label}_data.pkl", "rb") as f:
#             data_loaded = pickle.load(f)
#         extra_stats = data_loaded["extra_stats"]
#         hit_time_means = [np.mean(extra_stats["server_hit_by_cat"].get(int(cat), [0.0])) for cat in categories]
#         plt.bar(x + idx * width - width, hit_time_means, width, label=f"{case_label}")
#
#     plt.xticks(x, categories)
#     plt.xlabel("Server Category")
#     plt.ylabel("Average Hit Times")
#     plt.title("Average Server Hit Times by Category")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig("comparison_average_server_hit_times_by_category.png")
#     plt.close()
#
#     # # Server Hit Rate by Category (Mean)
#     # plt.figure(figsize=(10, 6))
#     # for idx, case_label in enumerate(case_labels):
#     #     with open(f"plot_data_cases/{case_label}_data.pkl", "rb") as f:
#     #         data_loaded = pickle.load(f)
#     #     extra_stats = data_loaded["extra_stats"]
#     #     hit_rate_means = [np.mean(extra_stats["server_hit_rate_by_cat"].get(int(cat), [0.0])) for cat in categories]
#     #     plt.bar(x + idx * width - width, hit_rate_means, width, label=f"{case_label}")
#     #
#     # plt.xticks(x, categories)
#     # plt.xlabel("Server Category")
#     # plt.ylabel("Average Hit Rate")
#     # plt.title("Average Server Hit Rate by Category")
#     # plt.legend()
#     # plt.grid(True)
#     # plt.savefig("comparison_average_server_hit_rate_by_category.png")
#     # plt.close()
#
#     #
#     # # User Local Reward by Environment (Mean)
#     # user_types = ['Entertainment', 'Home', 'Wild', 'Traffic', 'Work']
#     # x_types = np.arange(len(user_types))
#     # plt.figure(figsize=(12, 6))
#     # for idx, case_label in enumerate(case_labels):
#     #     with open(f"plot_data_cases/{case_label}_data.pkl", "rb") as f:
#     #         data_loaded = pickle.load(f)
#     #     extra_stats = data_loaded["extra_stats"]
#     #     local_means = [np.mean(extra_stats["user_local_by_env"].get(env, [0.0])) for env in user_types]
#     #     plt.bar(x_types + idx * width - width, local_means, width / 2, label=f"{case_label} Local", color=colors[idx % len(colors)])
#     #
#     #     idx+=1
#     #
#     # plt.xticks(x_types, user_types)
#     # plt.xlabel("User Environment")
#     # plt.ylabel("Average Reward")
#     # plt.title("Average User Local Reward by Environment")
#     # plt.legend()
#     # plt.grid(True)
#     # plt.savefig("comparison_average_user_rewards_by_env.png")
#     # plt.close()
#
#     # Server Service Quality (Mean)
#     plt.figure(figsize=(10, 6))
#     for idx, case_label in enumerate(case_labels):
#         with open(f"plot_data_cases/{case_label}_data.pkl", "rb") as f:
#             data_loaded = pickle.load(f)
#         extra_stats = data_loaded["extra_stats"]
#         service_quality = np.array([
#             np.mean((np.array(extra_stats["server_combined_by_cat"].get(int(cat), [0.0])) -
#             np.array(extra_stats["server_local_by_cat"].get(int(cat), [0.0])) * 0.5) / 0.5)
#             for cat in categories
#         ])
#         plt.bar(x + idx * width - width, service_quality, width, label=f"{case_label}")
#
#     plt.xticks(x, categories)
#     plt.xlabel("Server Category")
#     plt.ylabel("Average Service Quality")
#     plt.title("Average Server Service Quality by Category")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig("comparison_average_server_service_quality_by_category.png")
#     plt.close()
#
#     categories = ['0', '1', '2', '3', '4']
#
#     # Server computational saving (Mean)
#     plt.figure(figsize=(10, 6))
#     for idx, case_label in enumerate(case_labels):
#         load_path = f"plot_data_cases/{case_label}_data.pkl"
#         with open(load_path, "rb") as f:
#             data_loaded = pickle.load(f)
#         extra_stats = data_loaded["extra_stats"]
#         computational_saving = np.array([
#             np.mean(extra_stats["server_hit_by_uav"].get(int(cat), [0.0])) +
#             np.mean(extra_stats["server_sim_by_uav"].get(int(cat), [0.0]))
#             for cat in categories
#         ])
#         plt.bar(x + idx * width - width, computational_saving, width, label=f"{case_label}")
#
#     plt.xticks(x, categories)
#     plt.xlabel("Server Category")
#     plt.ylabel("Average Computational Saving")
#     plt.title("Average Server Computational Saving by Category")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig("comparison_average_server_computational_saving_by_category.png")
#     plt.close()

# np.mean(extra["server_queue_by_cat"].get(int(cat), [0.0])[-50:])
def plot_episode_comparisons_average(case_labels):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    # blue, orange, green, red
    # colors = ['#1f77b4', '#d62728', '#2ca02c']  # blue, red, green

    # colors = ['#1f77b4', '#2ca02c', '#d62728']  # blue, green, red
    # colors = ['#1f77b4', '#d62728', '#2ca02c']
    # colors = ['#1f77b4',  # blue
    #           '#ff7f0e',  # orange
    #           '#2ca02c',  # green
    #           '#d62728',  # red
    #           '#9467bd']  # purple

    categories = ['1', '2', '3', '4', '5']
    n_cases = len(case_labels)
    width = 0.25
    gap = width  # space between category groups
    group_width = n_cases * width

    # compute the left edge of each category block
    base_x = np.arange(len(categories)) * (group_width + gap)

    # ========== Server Queue Length ==========
    plt.figure(figsize=(10, 6))
    for idx, case_label in enumerate(case_labels):
        with open(f"plot_data_cases/{case_label}_data.pkl", "rb") as f:
            data_loaded = pickle.load(f)
        extra = data_loaded["extra_stats"]
        queue_means = [
            np.mean(extra["server_queue_by_cat"].get(int(cat), [0.0]))
            for cat in categories
        ]
        plt.bar(base_x + idx * width, queue_means, width, label=case_label, color=colors[idx % len(colors)])
    plt.xticks(base_x + group_width/2, categories)
    plt.xlabel("Server Category", fontsize=14)
    plt.ylabel("Average Queue Length", fontsize=14)
    # plt.title("Average Server Queue Length by Category")
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig("comparison_average_server_queue_length.png", dpi=300)
    plt.close()

    # ========== Server Combined Reward ==========
    plt.figure(figsize=(10, 6))
    for idx, case_label in enumerate(case_labels):
        with open(f"plot_data_cases/{case_label}_data.pkl", "rb") as f:
            data_loaded = pickle.load(f)
        extra = data_loaded["extra_stats"]
        reward_means = [
            np.mean(extra["server_combined_by_cat"].get(int(cat), [0.0]))
            for cat in categories
        ]
        plt.bar(base_x + idx * width, reward_means, width, label=case_label, color=colors[idx % len(colors)])
    plt.xticks(base_x + group_width/2, categories)
    plt.xlabel("Server Category", fontsize=14)
    plt.ylabel("Average Combined Reward", fontsize=14)
    # plt.title("Average Server Combined Reward by Category")
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig("comparison_average_server_combined_reward_by_category.png", dpi=300)
    plt.close()

    # ========== Server Hit Times ==========
    plt.figure(figsize=(10, 6))
    for idx, case_label in enumerate(case_labels):
        with open(f"plot_data_cases/{case_label}_data.pkl", "rb") as f:
            data_loaded = pickle.load(f)
        extra = data_loaded["extra_stats"]
        hit_means = [
            np.mean(extra["server_hit_by_cat"].get(int(cat), [0.0]))
            for cat in categories
        ]
        plt.bar(base_x + idx * width, hit_means, width, label=case_label, color=colors[idx % len(colors)])
    plt.xticks(base_x + group_width/2, categories)
    plt.xlabel("Server Category", fontsize=14)
    plt.ylabel("Average Hit Times", fontsize=14)
    # plt.title("Average Server Hit Times by Category")
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig("comparison_average_server_hit_times_by_category.png", dpi=300)
    plt.close()

    # # ========== Server Hit Rate ==========
    # plt.figure(figsize=(10, 6))
    # for idx, case_label in enumerate(case_labels):
    #     with open(f"plot_data_cases_old_0710/{case_label}_data.pkl", "rb") as f:
    #         data_loaded = pickle.load(f)
    #     extra = data_loaded["extra_stats"]
    #     hit_means = [
    #         np.mean(extra["server_hit_rate_by_cat"].get(int(cat), [0.0]))
    #         for cat in categories
    #     ]
    #     plt.bar(base_x + idx * width, hit_means, width, label=case_label, color=colors[idx % len(colors)])
    # plt.xticks(base_x + group_width/2, categories)
    # plt.xlabel("Server Category", fontsize=14)
    # plt.ylabel("Average Hit Rates", fontsize=14)
    # # plt.title("Average Server Hit Rates by Category")
    # plt.legend(fontsize=14)
    # plt.grid(True)
    # plt.savefig("comparison_average_server_hit_rates_by_category.png")
    # plt.close()

    # # ========== Server Hit Rate ==========
    # plt.figure(figsize=(10, 6))
    # for idx, case_label in enumerate(case_labels):
    #     with open(f"plot_data_cases_old_0710/{case_label}_data.pkl", "rb") as f:
    #         data_loaded = pickle.load(f)
    #     extra = data_loaded["extra_stats"]
    #     hit_means = [
    #         np.mean(extra["server_hit_rate_by_cat"].get(int(cat), [0.0]))
    #         for cat in categories
    #     ]
    #     plt.bar(base_x + idx * width, hit_means, width, label=case_label, color=colors[idx % len(colors)])
    # plt.xticks(base_x + group_width/2, categories)
    # plt.xlabel("Server Category", fontsize=14)
    # plt.ylabel("Average Hit Rates", fontsize=14)
    # # plt.title("Average Server Hit Rates by Category")
    # plt.legend(fontsize=14)
    # plt.grid(True)
    # plt.savefig("comparison_average_server_hit_rates_by_category.png")
    # plt.close()



    # ========== User Local Reward ==========
    user_types = ['Entertainment', 'Home', 'Wild', 'Traffic', 'Work']
    x_types = np.arange(len(user_types)) * (group_width + gap)
    plt.figure(figsize=(12, 6))
    for idx, case_label in enumerate(case_labels):
        with open(f"plot_data_cases/{case_label}_data.pkl", "rb") as f:
            data_loaded = pickle.load(f)
        extra = data_loaded["extra_stats"]
        local_means = [
            np.mean(extra["user_local_by_env"].get(env, [0.0]))
            for env in user_types
        ]
        plt.bar(x_types + idx * (width/2), local_means, width/2,
                label=f"{case_label} Local", color=colors[idx % len(colors)])
    plt.xticks(x_types + group_width/4, user_types)
    plt.xlabel("User Environment", fontsize=14)
    plt.ylabel("Average Reward", fontsize=14)
    # plt.title("Average User Local Reward by Environment")
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig("comparison_average_user_rewards_by_env.png", dpi=300)
    plt.close()

    # ========== Server Service Quality ==========
    plt.figure(figsize=(10, 6))
    for idx, case_label in enumerate(case_labels):
        with open(f"plot_data_cases/{case_label}_data.pkl", "rb") as f:
            data_loaded = pickle.load(f)
        extra = data_loaded["extra_stats"]
        sq = np.array([
            (np.mean(extra["server_combined_by_cat"].get(int(cat), [0.0])) -
             0.5 * np.mean(extra["server_local_by_cat"].get(int(cat), [0.0]))
            ) / 0.5
            for cat in categories
        ])
        plt.bar(base_x + idx * width, sq, width, label=case_label, color=colors[idx % len(colors)])
    plt.xticks(base_x + group_width/2, categories)
    plt.xlabel("Server Category", fontsize=14)
    plt.ylabel("Average Service Quality", fontsize=14)
    # plt.title("Average Server Service Quality by Category")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_average_server_service_quality_by_category.png", dpi=300)
    plt.close()

    # ========== Computational Saving ==========
    categories = ['0', '1', '2', '3', '4']
    plt.figure(figsize=(10, 6))
    for idx, case_label in enumerate(case_labels):
        with open(f"plot_data_cases/{case_label}_data.pkl", "rb") as f:
            data_loaded = pickle.load(f)
        extra = data_loaded["extra_stats"]
        saving = np.array([
            np.mean(extra["server_hit_by_uav"].get(int(cat), [0.0])) +
            np.mean(extra["server_sim_by_uav"].get(int(cat), [0.0]))
            for cat in categories
        ])
        plt.bar(base_x + idx * width, saving, width, label=case_label, color=colors[idx % len(colors)])
    plt.xticks(
        base_x + group_width / 2,
        [str(int(cat) + 1) for cat in categories]
    )
    plt.xlabel("Server Category", fontsize=14)
    plt.ylabel("Average Computational Saving", fontsize=14)
    # plt.title("Average Server Computational Saving by Category")
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig("comparison_average_server_computational_saving_by_category.png", dpi=300)
    plt.close()



    # # ========== Computational Energy Saving ==========
    # categories = ['0', '1', '2', '3', '4']
    #
    # # Corresponding divisors (FPS / processing rate)
    # divisors = np.array([67, 45, 56, 31, 22])
    #
    # # Rendering power = 1.7 W
    # power_watts = 1.7
    #
    # plt.figure(figsize=(10, 6))
    #
    # for idx, case_label in enumerate(case_labels):
    #     with open(f"plot_data_cases_old_0710/{case_label}_data.pkl", "rb") as f:
    #         data_loaded = pickle.load(f)
    #
    #     extra = data_loaded["extra_stats"]
    #
    #     # 1. Number of tasks
    #     task_counts = np.array([
    #         np.mean(extra["server_hit_by_uav"].get(int(cat), [0.0])) +
    #         np.mean(extra["server_sim_by_uav"].get(int(cat), [0.0]))
    #         for cat in categories
    #     ])
    #
    #     # 2. Energy saving (J)
    #     # Formula: (tasks / processing rate) * power
    #     # Units: (Tasks / (Tasks/sec)) * Watts = Seconds * Watts = Joules
    #     computed_energy_saving = (task_counts / divisors) * power_watts
    #
    #     # 3. Plot
    #     plt.bar(base_x + idx * width, computed_energy_saving, width, label=case_label, color=colors[idx % len(colors)])
    #
    # plt.xticks(
    #     base_x + group_width / 2,
    #     [str(int(cat) + 1) for cat in categories]
    # )
    #
    # plt.xlabel("Server Category", fontsize=14)
    # # Add the unit (J) to the y-axis label
    # plt.ylabel("Average Computational Energy Saving (Joules)", fontsize=14)
    # # plt.title("Average Server Computational Energy Saving by Category")
    #
    # plt.legend(fontsize=14)
    # plt.grid(True)
    # plt.savefig("comparison_average_server_computational_energy_saving_by_category.png")
    # plt.close()

    # ========== Figure style (optional, paper-friendly) ==========
    # plt.style.use('seaborn-v0_8-whitegrid') # recommended if your matplotlib version supports it
    # ========== Computational Time & Energy Saving (Dual Axis) ==========
    categories = ['0', '1', '2', '3', '4']
    divisors = np.array([67, 45, 56, 31, 22])  # computation capabilities ca_j of server categories 1-5
    power_watts = 1.7  # rendering power P^Ren = 1.7 W

    # Layout parameters
    num_cases = len(case_labels)
    width = 0.8 / num_cases
    base_x = np.arange(len(categories))
    group_width = width * num_cases

    # Create the figure
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # ========== Plot loop (left axis ax1) ==========
    for idx, case_label in enumerate(case_labels):
        with open(f"plot_data_cases/{case_label}_data.pkl", "rb") as f:
            data_loaded = pickle.load(f)

        extra = data_loaded["extra_stats"]

        # 1. Number of tasks
        task_counts = np.array([
            np.mean(extra["server_hit_by_uav"].get(int(cat), [0.0])) +
            np.mean(extra["server_sim_by_uav"].get(int(cat), [0.0]))
            for cat in categories
        ])

        # 2. Time saving -> left-axis data
        time_saving = task_counts / divisors

        # Plot in the original style (no edge, no alpha, default color list)
        ax1.bar(base_x + idx * width, time_saving, width,
                label=case_label,
                color=colors[idx % len(colors)])

    # ========== Left axis (time saving) ==========
    # X-axis ticks (original logic)
    ax1.set_xticks(base_x + group_width / 2)
    ax1.set_xticklabels([str(int(cat) + 1) for cat in categories])

    ax1.set_xlabel("Server Category", fontsize=14)
    ax1.set_ylabel("Average Computational Time Saving (s)", fontsize=14)
    ax1.grid(True)  # keep the original solid grid style
    ax1.legend(fontsize=14)

    # ========== Right axis (energy saving) ==========
    ax2 = ax1.twinx()

    # Key point: right-axis range = left-axis range * 1.7 W
    y1_min, y1_max = ax1.get_ylim()
    ax2.set_ylim(y1_min * power_watts, y1_max * power_watts)

    # Right-axis label
    ax2.set_ylabel("Average Computational Energy Saving (J)", fontsize=14)

    # No grid on the right axis to avoid overlapping grids
    ax2.grid(False)

    # Save
    plt.savefig("dual_axis_computational_saving.png", dpi=300)
    plt.close()


# def plot_hit_similarity_discount(case_label):
#     colors = ['#6CA6CD', '#FA8072']
#     load_path = f"plot_data_cases/{case_label}_data.pkl"
#     with open(load_path, "rb") as f:
#         data_loaded = pickle.load(f)
#     extra_stats = data_loaded["extra_stats"]
#     num_episodes = len(extra_stats["server_hit_by_uav"][0])
#     x_ep = np.arange(num_episodes) + 1
#
#     all_hit = np.mean([extra_stats["server_hit_by_uav"][uav] for uav in extra_stats["server_hit_by_uav"]], axis=0)
#     all_sim = np.mean([extra_stats["server_sim_by_uav"][uav] for uav in extra_stats["server_sim_by_uav"]], axis=0)
#
#     plt.figure(figsize=(12, 6))
#     plt.plot(x_ep, smooth_curve(all_hit), marker='o', linestyle='-', markevery=5, label="Overall Hit Discount", color=colors[0])
#     plt.plot(x_ep, smooth_curve(all_sim), marker='o', linestyle='--', markevery=5, label="Overall Similarity Discount", color=colors[1])
#     plt.xlabel("Episode")
#     plt.ylabel("Value")
#     plt.title(f"Overall Server Saving & Similarity Trends ({case_label})")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(f"{case_label}_overall_saving_similarity.png")
#     plt.close()

# def plot_hit_similarity_discount(case_label):
#     # colors = ['#6CA6CD', '#FA8072', '#9ACD32'] , color=colors[0]
#     colors = ['#1f77b4', '#d62728', '#2ca02c']
#     load_path = f"plot_data_cases_1_2/{case_label}_data.pkl"
#     with open(load_path, "rb") as f:
#         data_loaded = pickle.load(f)
#     extra_stats = data_loaded["extra_stats"]
#     num_episodes = len(extra_stats["server_hit_by_uav"][0])
#     x_ep = np.arange(num_episodes) + 1
#
#     all_hit = np.mean([extra_stats["server_hit_by_uav"][uav] for uav in extra_stats["server_hit_by_uav"]], axis=0)
#     all_sim = np.mean([extra_stats["server_sim_by_uav"][uav] for uav in extra_stats["server_sim_by_uav"]], axis=0)
#     total_discount = all_hit + all_sim
#
#     plt.figure(figsize=(12, 6))
#     plt.plot(x_ep, smooth_curve(all_hit), marker='o', linestyle='-', markevery=1, label="Overall Hit Saving", color=colors[0])
#     plt.plot(x_ep, smooth_curve(all_sim), marker='o', linestyle='--', markevery=1, label="Overall Similarity Saving", color=colors[1])
#     plt.plot(x_ep, smooth_curve(total_discount), marker='o', linestyle='-.', markevery=1, label="Total Saving by Caching", color=colors[2])
#
#     plt.xlabel("Episode")
#     plt.ylabel("Value")
#     plt.title(f"Overall Server Saving & Similarity Trends ({case_label})")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(f"{case_label}_overall_saving_similarity.png")
#     plt.close()

def plot_hit_similarity_discount(case_label):
    colors = ['#1f77b4', '#d62728', '#2ca02c']
    load_path = f"plot_data_cases/{case_label}_data.pkl"
    with open(load_path, "rb") as f:
        data_loaded = pickle.load(f)
    extra_stats = data_loaded["extra_stats"]
    num_episodes = len(extra_stats["server_hit_by_uav"][0])
    x_ep = np.arange(num_episodes) + 1

    all_hit = np.mean([extra_stats["server_hit_by_uav"][uav] for uav in extra_stats["server_hit_by_uav"]], axis=0)
    all_sim = np.mean([extra_stats["server_sim_by_uav"][uav] for uav in extra_stats["server_sim_by_uav"]], axis=0)
    total_discount = all_hit + all_sim

    # Piecewise smooth_curve
    hit_smoothed = smooth_curve(all_hit[:100]) + smooth_curve(all_hit[100:], window_size=15)
    sim_smoothed = smooth_curve(all_sim[:100]) + smooth_curve(all_sim[100:], window_size=15)
    total_smoothed = smooth_curve(total_discount[:100]) + smooth_curve(total_discount[100:], window_size=5)

    plt.figure(figsize=(12, 6))
    plt.plot(x_ep, hit_smoothed, marker='o', linestyle='-', markevery=1, label="Overall Hit Saving", color=colors[0])
    plt.plot(x_ep, sim_smoothed, marker='o', linestyle='--', markevery=1, label="Overall Similarity Saving", color=colors[1])
    plt.plot(x_ep, total_smoothed, marker='o', linestyle='-.', markevery=1, label="Total Saving by Caching", color=colors[2])

    plt.xlabel("Episode",fontsize=14)
    plt.ylabel("Value",fontsize=14)
    # plt.title(f"Overall Server Saving & Similarity Trends ({case_label})")
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig(f"{case_label}_overall_saving_similarity.png")
    plt.close()


# def plot_average_queue_length(case_labels):
#     categories = ['1', '2', '3', '4', '5']
#     colors = ['#1f77b4', '#d62728', '#2ca02c']
#     # colors = ['#6CA6CD', '#9ACD32', '#FA8072', '#FFB347', '#FFD966']
#     x = np.arange(len(categories))
#     width = 0.25
#
#     plt.figure(figsize=(10, 6))
#
#     for idx, case_label in enumerate(case_labels):
#         load_path = f"1_2_plot_data_cases/{case_label}_data.pkl"
#         with open(load_path, "rb") as f:
#             data_loaded = pickle.load(f)
#         extra_stats = data_loaded["extra_stats"]
#         queue_means = [np.mean(extra_stats["server_queue_by_cat"].get(int(cat), [0.0])[-50:]) for cat in categories]
#         plt.bar(x + idx * width - width, queue_means, width, label=f"{case_label}", color=colors[idx % len(colors)])
#
#     plt.xticks(x, categories)
#     plt.xlabel("Server Category")
#     plt.ylabel("Average Queue Length")
#     plt.title("Average Server Queue Length by Category")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig("comparison_average_server_queue_length.png")
#     plt.close()

#
def plot_service_quality_trends(case_labels):
    categories = ['1', '2', '3', '4', '5']
    colors = ['#6CA6CD', '#9ACD32', '#FA8072', '#FFB347', '#FFD966']

    plt.figure(figsize=(12, 7))

    for idx, case_label in enumerate(case_labels):
        load_path = f"plot_data_cases/{case_label}_data30.pkl"
        with open(load_path, "rb") as f:
            data_loaded = pickle.load(f)

        combined_by_cat = data_loaded["extra_stats"]["server_combined_by_cat"]
        local_by_cat = data_loaded["extra_stats"]["server_local_by_cat"]
        service_quality_over_time = {cat: [] for cat in categories}

        for cat in categories:
            combined_values = np.array(combined_by_cat.get(int(cat), [0.0]))
            local_values = np.array(local_by_cat.get(int(cat), [0.0]))
            quality_values = (combined_values - local_values * 0.5) / 0.5
            smoothed_quality_values = smooth_curve(quality_values)
            service_quality_over_time[cat].extend(smoothed_quality_values)

        for cat_idx, cat in enumerate(categories):
            plt.plot(service_quality_over_time[cat], label=f"Case {case_label} - Category {cat}",
                     color=colors[cat_idx], linestyle='-', marker='o')

    plt.xlabel("Data Points")
    plt.ylabel("Service Quality")
    plt.title("Server Service Quality Trends by Category and Case")
    plt.legend()
    plt.grid(True)
    plt.savefig("server_service_quality_trends.png")
    plt.close()


def plot_service_quality_boxplot(case_labels):
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle

    categories = ['1', '2', '3', '4', '5']
    data_to_plot = []
    labels = []

    for cat in categories:
        for case_label in case_labels:
            load_path = f"1_2_plot_data_cases/{case_label}_data.pkl"
            with open(load_path, "rb") as f:
                data_loaded = pickle.load(f)

            combined_by_cat = data_loaded["extra_stats"]["server_combined_by_cat"]
            local_by_cat = data_loaded["extra_stats"]["server_local_by_cat"]

            combined_values = np.array(combined_by_cat.get(int(cat), [0.0]))
            local_values = np.array(local_by_cat.get(int(cat), [0.0]))
            quality_values = (combined_values - local_values * 0.5) / 0.5
            data_to_plot.append(quality_values)
            labels.append(f"Cat {cat} - Case {case_label}")

    plt.figure(figsize=(14, 8))
    plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
    plt.xticks(rotation=45)
    plt.xlabel("Server Category and Case")
    plt.ylabel("Service Quality")
    plt.title("Server Service Quality Distribution by Category and Case")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("server_service_quality_boxplot.png")
    plt.close()

def plot_service_quality_barplot(case_labels):
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle

    categories = ['1', '2', '3', '4', '5']
    width = 0.15
    gap = 0.05
    n_cases = len(case_labels)
    group_width = (width * len(categories)) + gap

    base_x = np.arange(n_cases) * group_width

    plt.figure(figsize=(14, 8))

    for idx, cat in enumerate(categories):
        means = []
        for case_label in case_labels:
            load_path = f"1_2_plot_data_cases/{case_label}_data.pkl"
            with open(load_path, "rb") as f:
                data_loaded = pickle.load(f)

            combined_by_cat = data_loaded["extra_stats"]["server_combined_by_cat"]
            local_by_cat = data_loaded["extra_stats"]["server_local_by_cat"]

            combined_values = np.array(combined_by_cat.get(int(cat), [0.0]))
            local_values = np.array(local_by_cat.get(int(cat), [0.0]))
            quality_values = (combined_values - local_values * 0.5) / 0.5
            means.append(np.mean(quality_values))

        plt.bar(base_x + idx * width, means, width=width, label=f"Category {cat}")

    plt.xticks(base_x + (width * (len(categories) - 1) / 2), case_labels)
    plt.xlabel("Case Labels")
    plt.ylabel("Mean Service Quality")
    plt.title("Mean Server Service Quality by Case and Category")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("server_service_quality_barplot.png")
    plt.close()


def plot_server_feature_heatmap(case_label, NUM_UAVS=5):

    load_path = f"plot_data_cases/{case_label}_data.pkl"
    with open(load_path, "rb") as f:
        data_loaded = pickle.load(f)

    data = data_loaded["extra_stats"]["server_feature_history"]

    mat = np.vstack([np.mean(np.vstack(data[j]), axis=0)[:100] for j in range(NUM_UAVS)])


    plt.figure(figsize=(16,6))

    # sns.heatmap(mat, cmap="viridis")

    # Label the y-ticks 1..NUM_UAVS (the default would be 0..NUM_UAVS-1)
    ax = sns.heatmap(mat, cmap="viridis", yticklabels=np.arange(1, mat.shape[0] + 1))

    plt.xlabel("Content Index", fontsize=14)
    plt.ylabel("Server Category", fontsize=14)
    # plt.title(f"Server Feature Heatmap ({case_label})")
    plt.tight_layout()
    plt.show()
    plt.savefig("server_feature_heatmap.png")
    plt.close()

    # plt.figure(figsize=(12, 4))
    # plt.imshow(mat, aspect='auto', cmap='viridis')
    # plt.colorbar(label='Mean Feature')
    # plt.xlabel('Feature Index (first 100)')
    # plt.ylabel('Server Index')
    # plt.title(f'Server Feature Heatmap ({case_label})')
    # plt.tight_layout()
    # plt.show()



def plot_first_episode_comparisons(case_labels):
    categories = ['1', '2', '3', '4', '5']
    x = np.arange(len(categories))
    width = 0.25

    # Server Queue Length
    plt.figure(figsize=(10, 6))
    for idx, case_label in enumerate(case_labels):
        load_path = f"plot_data_cases/{case_label}_data.pkl"
        with open(load_path, "rb") as f:
            data_loaded = pickle.load(f)
        extra_stats = data_loaded["extra_stats"]
        queue_means = [extra_stats["server_queue_by_cat"].get(int(cat), [0.0])[0] for cat in categories]
        plt.bar(x + idx * width - width, queue_means, width, label=f"{case_label}")

    plt.xticks(x, categories)
    plt.xlabel("Server Category")
    plt.ylabel("Queue Length")
    plt.title("Server Queue Length by Category")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_server_queue_length.png")
    plt.close()

    # Server Combined Reward
    plt.figure(figsize=(10, 6))
    for idx, case_label in enumerate(case_labels):
        load_path = f"plot_data_cases/{case_label}_data.pkl"
        with open(load_path, "rb") as f:
            data_loaded = pickle.load(f)
        extra_stats = data_loaded["extra_stats"]
        reward_means = [extra_stats["server_combined_by_cat"].get(int(cat), [0.0])[0] for cat in categories]
        plt.bar(x + idx * width - width, reward_means, width, label=f"{case_label}")

    plt.xticks(x, categories)
    plt.xlabel("Server Category")
    plt.ylabel("Combined Reward")
    plt.title("Server Combined Reward by Category")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_server_combined_reward_by_category.png")
    plt.close()

    # Server Hit Rate by Category
    plt.figure(figsize=(10, 6))
    for idx, case_label in enumerate(case_labels):
        load_path = f"plot_data_cases/{case_label}_data.pkl"
        with open(load_path, "rb") as f:
            data_loaded = pickle.load(f)
        extra_stats = data_loaded["extra_stats"]
        hit_rate_means = [extra_stats["server_hit_by_cat"].get(int(cat), [0.0])[0] for cat in categories]
        plt.bar(x + idx * width - width, hit_rate_means, width, label=f"{case_label}")

    plt.xticks(x, categories)
    plt.xlabel("Server Category")
    plt.ylabel("Hit Rate")
    plt.title("Server Hit Rate by Category")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_server_hit_rate_by_category.png")
    plt.close()

    # User Local and Combined Reward
    user_types = ['Entertainment', 'Home', 'Wild', 'Traffic', 'Work']
    x_types = np.arange(len(user_types))
    plt.figure(figsize=(12, 6))
    for idx, case_label in enumerate(case_labels):
        load_path = f"plot_data_cases/{case_label}_data.pkl"
        with open(load_path, "rb") as f:
            data_loaded = pickle.load(f)
        extra_stats = data_loaded["extra_stats"]
        local_means = [extra_stats["user_local_by_env"].get(env, [0.0])[0] for env in user_types]
        # global_means = [extra_stats["user_combined_by_env"].get(env, [0.0])[-1] for env in user_types]
        plt.bar(x_types + idx * width - width, local_means, width / 2, label=f"{case_label} Local")
        # plt.bar(x_types + idx * width - width / 2, global_means, width / 2, label=f"{case_label} Combined")

    plt.xticks(x_types, ['1', '2', '3', '4', '5'])
    plt.xlabel("User Environment")
    plt.ylabel("Reward")
    plt.title("User Local  Reward by Environment")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_user_rewards_by_env.png")
    plt.close()

    # Server service quality
    plt.figure(figsize=(10, 6))
    for idx, case_label in enumerate(case_labels):
        load_path = f"plot_data_cases/{case_label}_data.pkl"
        with open(load_path, "rb") as f:
            data_loaded = pickle.load(f)
        extra_stats = data_loaded["extra_stats"]
        service_quality = np.array([
            (extra_stats["server_combined_by_cat"].get(int(cat), [0.0])[0] -
            extra_stats["server_local_by_cat"].get(int(cat), [0.0])[0] * 0.5) / 0.5
            for cat in categories
        ])
        plt.bar(x + idx * width - width, service_quality, width, label=f"{case_label}")

    plt.xticks(x, categories)
    plt.xlabel("Server Category")
    plt.ylabel("service_quality")
    plt.title("Server service_quality by Category")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_server_service_quality_by_category.png")
    plt.close()

    categories = ['0', '1', '2', '3', '4']
    # Server computational_saving
    plt.figure(figsize=(10, 6))
    for idx, case_label in enumerate(case_labels):
        load_path = f"plot_data_cases_old_0710/{case_label}_data.pkl"
        with open(load_path, "rb") as f:
            data_loaded = pickle.load(f)
        extra_stats = data_loaded["extra_stats"]
        computational_saving = np.array([
            extra_stats["server_hit_by_uav"].get(int(cat), [0.0])[0] +
            extra_stats["server_sim_by_uav"].get(int(cat), [0.0])[0]
            for cat in categories
        ])
        plt.bar(x + idx * width - width, computational_saving, width, label=f"{case_label}")

    plt.xticks(x, categories)
    plt.xlabel("Server Category")
    plt.ylabel("computational_saving")
    plt.title("Server computational_saving by Category")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_server_computational_saving_by_category.png")
    plt.close()

def plot_first_15_episode_comparisons_average(case_labels):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    # blue, orange, green, red
    # colors = ['#1f77b4', '#d62728', '#2ca02c']  # blue, red, green

    # colors = ['#1f77b4', '#2ca02c', '#d62728']  # blue, green, red
    # colors = ['#1f77b4', '#d62728', '#2ca02c']
    # colors = ['#1f77b4',  # blue
    #           '#ff7f0e',  # orange
    #           '#2ca02c',  # green
    #           '#d62728',  # red
    #           '#9467bd']  # purple
    n_1 = 0
    n_2 = 99
    categories = ['1', '2', '3', '4', '5']
    n_cases = len(case_labels)
    width = 0.25
    gap = width  # space between category groups
    group_width = n_cases * width

    # compute the left edge of each category block
    base_x = np.arange(len(categories)) * (group_width + gap)

    # ========== Server Queue Length ==========
    plt.figure(figsize=(10, 6))
    for idx, case_label in enumerate(case_labels):
        with open(f"plot_data_cases/{case_label}_data.pkl", "rb") as f:
            data_loaded = pickle.load(f)
        extra = data_loaded["extra_stats"]
        queue_means = [
            np.mean(extra["server_queue_by_cat"].get(int(cat), [0.0])[n_1:n_2])
            for cat in categories
        ]
        plt.bar(base_x + idx * width, queue_means, width, label=case_label, color=colors[idx % len(colors)])
    plt.xticks(base_x + group_width / 2, categories)
    plt.xlabel("Server Category", fontsize=14)
    plt.ylabel("Average Queue Length", fontsize=14)
    # plt.title("Average Server Queue Length by Category")
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig("comparison_average_server_queue_length.png")
    plt.close()

    # ========== Server Combined Reward ==========
    plt.figure(figsize=(10, 6))
    for idx, case_label in enumerate(case_labels):
        with open(f"plot_data_cases/{case_label}_data.pkl", "rb") as f:
            data_loaded = pickle.load(f)
        extra = data_loaded["extra_stats"]
        reward_means = [
            np.mean(extra["server_combined_by_cat"].get(int(cat), [0.0])[n_1:n_2])
            for cat in categories
        ]
        plt.bar(base_x + idx * width, reward_means, width, label=case_label, color=colors[idx % len(colors)])
    plt.xticks(base_x + group_width / 2, categories)
    plt.xlabel("Server Category", fontsize=14)
    plt.ylabel("Average Combined Reward", fontsize=14)
    # plt.title("Average Server Combined Reward by Category")
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig("comparison_average_server_combined_reward_by_category.png")
    plt.close()

    # ========== Server Hit Times ==========
    plt.figure(figsize=(10, 6))
    for idx, case_label in enumerate(case_labels):
        with open(f"plot_data_cases/{case_label}_data.pkl", "rb") as f:
            data_loaded = pickle.load(f)
        extra = data_loaded["extra_stats"]
        hit_means = [
            np.mean(extra["server_hit_by_cat"].get(int(cat), [0.0])[n_1:n_2])
            for cat in categories
        ]
        plt.bar(base_x + idx * width, hit_means, width, label=case_label, color=colors[idx % len(colors)])
    plt.xticks(base_x + group_width / 2, categories)
    plt.xlabel("Server Category", fontsize=14)
    plt.ylabel("Average Hit Times", fontsize=14)
    # plt.title("Average Server Hit Times by Category")
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig("comparison_average_server_hit_times_by_category.png")
    plt.close()

    # # ========== Server Hit Rate ==========
    # plt.figure(figsize=(10, 6))
    # for idx, case_label in enumerate(case_labels):
    #     with open(f"plot_data_cases/{case_label}_data.pkl", "rb") as f:
    #         data_loaded = pickle.load(f)
    #     extra = data_loaded["extra_stats"]
    #     hit_means = [
    #         np.mean(extra["server_hit_rate_by_cat"].get(int(cat), [0.0])[0:15])
    #         for cat in categories
    #     ]
    #     plt.bar(base_x + idx * width, hit_means, width, label=case_label, color=colors[idx % len(colors)])
    # plt.xticks(base_x + group_width / 2, categories)
    # plt.xlabel("Server Category", fontsize=14)
    # plt.ylabel("Average Hit Rates", fontsize=14)
    # # plt.title("Average Server Hit Rates by Category")
    # plt.legend(fontsize=14)
    # plt.grid(True)
    # plt.savefig("comparison_average_server_hit_rates_by_category.png")
    # plt.close()

    # # ========== Server Hit Rate ==========
    # plt.figure(figsize=(10, 6))
    # for idx, case_label in enumerate(case_labels):
    #     with open(f"plot_data_cases/{case_label}_data.pkl", "rb") as f:
    #         data_loaded = pickle.load(f)
    #     extra = data_loaded["extra_stats"]
    #     hit_means = [
    #         np.mean(extra["server_hit_rate_by_cat"].get(int(cat), [0.0])[0:15])
    #         for cat in categories
    #     ]
    #     plt.bar(base_x + idx * width, hit_means, width, label=case_label, color=colors[idx % len(colors)])
    # plt.xticks(base_x + group_width / 2, categories)
    # plt.xlabel("Server Category", fontsize=14)
    # plt.ylabel("Average Hit Rates", fontsize=14)
    # # plt.title("Average Server Hit Rates by Category")
    # plt.legend(fontsize=14)
    # plt.grid(True)
    # plt.savefig("comparison_average_server_hit_rates_by_category.png")
    # plt.close()

    # ========== User Local Reward ==========
    user_types = ['Entertainment', 'Home', 'Wild', 'Traffic', 'Work']
    x_types = np.arange(len(user_types)) * (group_width + gap)
    plt.figure(figsize=(12, 6))
    for idx, case_label in enumerate(case_labels):
        with open(f"plot_data_cases/{case_label}_data.pkl", "rb") as f:
            data_loaded = pickle.load(f)
        extra = data_loaded["extra_stats"]
        local_means = [
            np.mean(extra["user_local_by_env"].get(env, [0.0])[n_1:n_2])
            for env in user_types
        ]
        plt.bar(x_types + idx * (width / 2), local_means, width / 2,
                label=f"{case_label} Local", color=colors[idx % len(colors)])
    plt.xticks(x_types + group_width / 4, user_types)
    plt.xlabel("User Environment", fontsize=14)
    plt.ylabel("Average Reward", fontsize=14)
    # plt.title("Average User Local Reward by Environment")
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig("comparison_average_user_rewards_by_env.png")
    plt.close()

    # ========== Server Service Quality ==========
    plt.figure(figsize=(10, 6))
    for idx, case_label in enumerate(case_labels):
        with open(f"plot_data_cases/{case_label}_data.pkl", "rb") as f:
            data_loaded = pickle.load(f)
        extra = data_loaded["extra_stats"]
        sq = np.array([
            (np.mean(extra["server_combined_by_cat"].get(int(cat), [0.0])[n_1:n_2]) -
             0.5 * np.mean(extra["server_local_by_cat"].get(int(cat), [0.0])[n_1:n_2])
             ) / 0.5
            for cat in categories
        ])
        plt.bar(base_x + idx * width, sq, width, label=case_label, color=colors[idx % len(colors)])
    plt.xticks(base_x + group_width / 2, categories)
    plt.xlabel("Server Category", fontsize=14)
    plt.ylabel("Average Service Quality", fontsize=14)
    # plt.title("Average Server Service Quality by Category")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_average_server_service_quality_by_category.png")
    plt.close()

    # ========== Computational Saving ==========
    categories = ['0', '1', '2', '3', '4']
    plt.figure(figsize=(10, 6))
    for idx, case_label in enumerate(case_labels):
        with open(f"plot_data_cases/{case_label}_data.pkl", "rb") as f:
            data_loaded = pickle.load(f)
        extra = data_loaded["extra_stats"]
        saving = np.array([
            np.mean(extra["server_hit_by_uav"].get(int(cat), [0.0])[n_1:n_2]) +
            np.mean(extra["server_sim_by_uav"].get(int(cat), [0.0])[n_1:n_2])
            for cat in categories
        ])
        plt.bar(base_x + idx * width, saving, width, label=case_label, color=colors[idx % len(colors)])
    plt.xticks(
        base_x + group_width / 2,
        [str(int(cat) + 1) for cat in categories]
    )
    plt.xlabel("Server Category", fontsize=14)
    plt.ylabel("Average Computational Saving", fontsize=14)
    # plt.title("Average Server Computational Saving by Category")
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig("comparison_average_server_computational_saving_by_category.png")
    plt.close()


 # colors = ['#1f77b4', '#2ca02c', '#d62728']
# Saving breakdown per case (cases side by side per category; hit at the bottom, hatched reuse on top)
def plot_saving_breakdown(case_labels, n1=0,n2=100):
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle

    categories = ['0', '1', '2', '3', '4']
    n_cases = len(case_labels)
    width = 0.75 / n_cases  # total bar width divided among cases
    hatch = '//'  # pattern for reuse portion
    # distinct color per case
    # case_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:n_cases]
    case_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:n_cases]

    gap = 0.75 / n_cases  # space between category groups
    group_width = n_cases * width

    # compute the left edge of each category block
    base_x = np.arange(len(categories)) * (group_width + gap)



    # Prepare data: rows=categories, cols=cases
    hit_data = np.zeros((len(categories), n_cases))
    reuse_data = np.zeros((len(categories), n_cases))
    for j, case_label in enumerate(case_labels):
        load_path = f"plot_data_cases/{case_label}_data.pkl"
        with open(load_path, 'rb') as f:
            data_loaded = pickle.load(f)
        extra = data_loaded['extra_stats']
        for i, cat in enumerate(categories):
            # hit counts
            hit_vals = extra['server_hit_by_uav'].get(int(cat), [0.0])[n1:n2]
            hit_data[i, j] = np.mean(hit_vals)
            # saving = hit_by_uav + sim_by_uav
            uav_hit = extra['server_hit_by_uav'].get(int(cat), [0.0])[n1:n2]
            uav_sim = extra['server_sim_by_uav'].get(int(cat), [0.0])[n1:n2]
            saving_vals = np.array(uav_hit) + np.array(uav_sim)
            # reuse portion is saving minus hit
            reuse_data[i, j] = np.mean(saving_vals) - hit_data[i, j]

    # Plot
    x = np.arange(len(categories))
    plt.figure(figsize=(12, 6))
    for j in range(n_cases):
        offsets = x + (j - (n_cases - 1) / 2) * width
        for i in range(len(categories)):
            # Hit bar (bottom)
            plt.bar(offsets[i], hit_data[i, j], width=width,
                    color=case_colors[j], edgecolor='black')
            # Reuse bar (hatched top)
            plt.bar(offsets[i], reuse_data[i, j], width=width,
                    bottom=hit_data[i, j], color=case_colors[j], edgecolor='black', hatch=hatch)

    # Case legend
    case_patches = [plt.matplotlib.patches.Patch(color=case_colors[j], label=f"Case {case_labels[j]}")
                    for j in range(n_cases)]
    # Hit vs Reuse legend
    hit_patch = plt.matplotlib.patches.Patch(facecolor='white', edgecolor='black', label='Hit Saving')
    reuse_patch = plt.matplotlib.patches.Patch(facecolor='white', edgecolor='black', hatch=hatch, label='Reuse Saving')

    plt.xticks(
        base_x,
        [str(int(cat) + 1) for cat in categories]
    )
    plt.xlabel('Server Category', fontsize=14)
    plt.ylabel('Saving Value', fontsize=14)
    # plt.title('Saving Breakdown by Case and Category')
    # add legends
    first_legend = plt.legend(handles=case_patches, title='Cases', loc='upper right', fontsize=14)
    plt.gca().add_artist(first_legend)
    plt.legend(handles=[hit_patch, reuse_patch], title='Portion', loc='center right', fontsize=14)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig('saving_breakdown_by_case_category.png')
    plt.close()

# # Saving breakdown per case (cases side by side per category; hit at the bottom, hatched reuse on top)
# def plot_saving_breakdown(case_labels, num_episodes=15):
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import pickle
#
#     categories = ['1', '2', '3', '4', '5']
#     n_cases = len(case_labels)
#     width = 0.8 / n_cases  # total bar width divided among cases
#     hatch = '//'  # pattern for reuse portion
#     # distinct color per case
#     #     # case_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:n_cases]
#     case_colors = ['#1f77b4', '#2ca02c', '#d62728'][:n_cases]
#
#     # Prepare data: rows=categories, cols=cases
#     hit_data = np.zeros((len(categories), n_cases))
#     reuse_data = np.zeros((len(categories), n_cases))
#     for j, case_label in enumerate(case_labels):
#         load_path = f"plot_data_cases_ELLA/{case_label}_data.pkl"
#         with open(load_path, 'rb') as f:
#             data_loaded = pickle.load(f)
#         extra = data_loaded['extra_stats']
#         for i, cat in enumerate(categories):
#             # hit counts
#             hit_vals = extra['server_hit_by_cat'].get(int(cat), [0.0])[:num_episodes]
#             hit_data[i, j] = np.mean(hit_vals)
#             # saving = hit_by_uav + sim_by_uav
#             uav_hit = extra['server_hit_by_uav'].get(int(cat), [0.0])[:num_episodes]
#             uav_sim = extra['server_sim_by_uav'].get(int(cat), [0.0])[:num_episodes]
#             saving_vals = np.array(uav_hit) + np.array(uav_sim)
#             # reuse portion is saving minus hit
#             reuse_data[i, j] = np.mean(saving_vals) - hit_data[i, j]
#
#     # Plot
#     x = np.arange(len(categories))
#     plt.figure(figsize=(10, 6))
#     for j in range(n_cases):
#         # center bars around each category tick
#         offsets = x + (j - (n_cases - 1) / 2) * width
#         for i in range(len(categories)):
#             # Hit bar (bottom)
#             plt.bar(offsets[i], hit_data[i, j], width=width,
#                     color=case_colors[j], edgecolor='black')
#             # Reuse bar (hatched top)
#             plt.bar(offsets[i], reuse_data[i, j], width=width,
#                     bottom=hit_data[i, j], color=case_colors[j], edgecolor='black', hatch=hatch)
#
#     # Case legend
#     case_patches = [plt.matplotlib.patches.Patch(color=case_colors[j], label=f"Case {case_labels[j]}")
#                     for j in range(n_cases)]
#     # Hit vs Reuse legend
#     hit_patch = plt.matplotlib.patches.Patch(facecolor='white', edgecolor='black', label='Hit Saving')
#     reuse_patch = plt.matplotlib.patches.Patch(facecolor='white', edgecolor='black', hatch=hatch, label='Reuse Saving')
#
#     plt.xticks(x, categories)
#     plt.xlabel('Server Category', fontsize=14)
#     plt.ylabel('Saving Value', fontsize=14)
#     plt.title('Saving Breakdown by Case and Category')
#     first_legend = plt.legend(handles=case_patches, title='Cases', loc='upper right')
#     plt.gca().add_artist(first_legend)
#     plt.legend(handles=[hit_patch, reuse_patch], title='Portion', loc='upper left')
#
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig('saving_breakdown_by_case_category.png')
#     plt.close()



if __name__ == '__main__':
    # cases = ["Hit+Similarity_local_learner_5", "Hit+Similarity_local_learner_10","Hit+Similarity_local_learner_15","Hit+Similarity_local_learner_20","Hit+Similarity_local_learner_25","Hit+Similarity_local_learner_30"]
    # cases = ["Only_Hit_local_learner", "Hit+Similarity_local_learner"]
    # cases = "Hit+Similarity_base"
    # cases = "Hit+Similarity_local_learner"

    # cases = ["Only_Hit_local_learner_15"]

    # cases = ["Hit+Similarity_test"]
    # cases = ["Only_Hit_local_learner", "Only_Hit_transfer", "Only_Hit_DC-ELLA", "Hit+Similarity_transfer", "Hit+Similarity_local_learner", "Hit+Similarity_DC-ELLA"]

    # cases = ["Hit+Similarity_local_learner", "Hit+Similarity_PG-ELLA", "Hit+Similarity_transfer", "Hit+Similarity_DC-ELLA"]
    cases = ["Hit+Similarity_local_learner", "Hit+Similarity_transfer",
             "Hit+Similarity_DC-ELLA"]
    # cases = ["Hit+Similarity_test", "Hit+Similarity_base_test"]
    # cases = ["Only_Hit_base_test", "Hit+Similarity_base_test", "No_Cache_base_test"]
    # cases = ["Only_Hit", "Hit+Similarity", "Hit+Similarity+LL"]
    # plot_critic_server_loss(cases)
    plot_server_combined_rewards(cases)
    # plot_overall_discount(cases)
    # plot_episode_comparisons_average(cases)
    # plot_saving_breakdown(cases, 0, 20)

    # # # plot_critic_user_loss(cases)
    # # # plot_user_local_rewards(cases)
    # # # plot_user_global_rewards(cases)
    # # # plot_server_global_rewards(cases)

    # # plot_final_episode_comparisons(cases)
    # # plot_hit_similarity_discount("Hit+Similarity_base_test")
    # #
    # # plot_hit_similarity_discount("Hit+Similarity_local_learner")
    # # plot_average_queue_length(cases)

    # plot_service_quality_trends(cases)
    # # plot_service_quality_barplot(cases)
    # plot_server_feature_heatmap(cases, NUM_UAVS=5)
    # plot_first_episode_comparisons(cases)

    # plot_first_15_episode_comparisons_average(cases)
    #


    print("All specified comparison plots generated successfully.")
