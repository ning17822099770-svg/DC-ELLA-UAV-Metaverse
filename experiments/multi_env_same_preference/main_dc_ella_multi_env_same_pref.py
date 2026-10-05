"""DC-ELLA in a dynamic environment with similar user preferences (Fig. 8).

The 100-episode test phase steps through the 4 configurations in
TEST_PARAMS_LIST, which keep the semantic preferences but change the task load
/ arrival rates; the environment switches every ENV_SWITCH_EPISODES = 25
episodes. At each switch DC-ELLA re-initializes the servers' policies from L
and D. Compared against the PPO local learner and transfer learning.

Entry point: parallel_compare_conditions().
Run from this folder:  python main_dc_ella_multi_env_same_pref.py

Paper: "Deep Lifelong Learning for Adaptive Semantic-Aware Content Reuse in
UAV-Assisted Metaverse", IEEE Transactions on Mobile Computing.
Equation / section numbers refer to the paper.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sympy import false
from torch.distributions import Bernoulli, Categorical
import matplotlib.pyplot as plt
import math
from math import floor, log2, exp, pi
import random
import pickle
import scipy.io
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing
# Use a non-interactive backend to avoid figure conflicts across processes
import matplotlib
import os
import argparse

import threading
#just user local
matplotlib.use('Agg')

# from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

from Server_LL_Learner_with_dictionary import ServerLifelongLearner

from Environments_new_feature import MultiUserEnv

from User_agent import MAPPOAgent

from network_define import ServerActorNetNoGRUForELLA, server_GRUCriticNet

from Server_agent import ServerMAPPOAgent


#############################################
# Set all random seeds (makes runs comparable)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
#############################################

###################################################
# ========== Global parameters and user/server initialization =========
###################################################

# Load image_label_mapping (image -> semantic labels) from pickle
with open("image_label_mapping.pkl", "rb") as f:
    image_label_mapping = pickle.load(f)
print("image_label_mapping loaded:", image_label_mapping)

# Load subject_similarity_matrix.mat (MPNet cosine similarities)
subject_data = scipy.io.loadmat('subject_similarity_matrix.mat')
subject_similarity_matrix = subject_data['subject_similarity_matrix']

# Load object_similarity_matrix.mat (MPNet cosine similarities)
object_data = scipy.io.loadmat('object_similarity_matrix.mat')
object_similarity_matrix = object_data['object_similarity_matrix']

NUM_USERS = 15
NUM_UAVS  = 5
UAV_HEIGHT = 50

P_USER = 0.1
P_UAV   = 0.5
G_USER  = 1
G_UAV   = 1
CARRIER_FREQ = 2.4e9
NOISE_POWER  = 10 ** (-20.4)
BIT_PER_CYCLE= 0.25
mu_downlink  = 8
mu_compute   = 4
MAPSIZE      = 1000

TASK_SIZE = 1e6
COMPRESSION_RATE = 16

task_rates = {
    'Entertainment': 160,
    'Home': 120,
    'Wild': 80,
    'Traffic': 200,
    'Work': 240
}

task_prob = {
    'Entertainment': 0.6,
    'Home': 0.5,
    'Wild': 0.4,
    'Traffic': 0.7,
    'Work': 0.8
}

channel_params = {
    'Entertainment': [12.08, 0.11, 1.6, 23],
    'Home': [4.88, 0.43, 0.1, 21],
    'Wild': [27.23, 0.08, 2.3, 34],
    'Traffic':    [9.61, 0.16, 1.0, 20],
    'Work': [9.61, 0.16, 1.0, 20],
}

bandwidth_map_user = {
    'Entertainment': 15e6,
    'Home': 10e6,
    'Wild': 5e6,
    'Traffic': 20e6,
    'Work': 25e6,
}

##################################################
# ========== Pretraining environment parameters (lambda^pre in Table II) ==========
##################################################
PRETRAIN_PARAMS = {  # environment config 1
        "NUM_USERS": 15,
        "USER_TYPES": ['Entertainment', 'Home', 'Wild', 'Traffic', 'Work'],
        "task_rates": {
            'Entertainment': 80,
            'Home': 60,
            'Wild': 40,
            'Traffic': 100,
            'Work': 120
        },
        "task_prob": {
            'Entertainment': 0.6,
            'Home': 0.5,
            'Wild': 0.4,
            'Traffic': 0.7,
            'Work': 0.8
        },
        "steps_per_episode": 30,
    }


##################################################
# ========== Test environment parameters (lambda^test in Table II) ==========
##################################################
# TEST_PARAMS = {
#     "NUM_USERS": 25,           # original number of users
#     "USER_TYPES": ['Entertainment', 'Home', 'Wild', 'Traffic', 'Work'],  # all environment types
#     "task_rates": {
#         'Entertainment': 160,
#         'Home': 120,
#         'Wild': 80,
#         'Traffic': 200,
#         'Work': 240
#     },
#     "task_prob": {
#         'Entertainment': 0.6,
#         'Home': 0.5,
#         'Wild': 0.4,
#         'Traffic': 0.7,
#         'Work': 0.8
#     },
#     "steps_per_episode": 30,    # original steps per episode
# }

# TEST_PARAMS = {
#     "NUM_USERS": 15,           # original number of users
#     "USER_TYPES": ['Entertainment', 'Home', 'Wild', 'Traffic', 'Work'],  # all environment types
#     "task_rates": {
#         'Entertainment': 140,
#         'Home': 120,
#         'Wild': 60,
#         'Traffic': 160,
#         'Work': 180
#     },
#     "task_prob": {
#         'Entertainment': 0.5,
#         'Home': 0.4,
#         'Wild': 0.3,
#         'Traffic': 0.6,
#         'Work': 0.7
#     },
#     "steps_per_episode": 30,    # original steps per episode
# }

TEST_PARAMS =      {  # environment config 1
        "NUM_USERS": 15,
        "USER_TYPES": ['Entertainment', 'Home', 'Wild', 'Traffic', 'Work'],
        "task_rates": {
            'Entertainment': 100,
            'Home': 30,
            'Wild': 20,
            'Traffic': 200,
            'Work': 120
        },
        "task_prob": {
            'Entertainment': 0.6,
            'Home': 0.5,
            'Wild': 0.4,
            'Traffic': 0.7,
            'Work': 0.8
        },
        "steps_per_episode": 30,
    }

# {  # environment config 5 (= config 1)
#     "NUM_USERS": 15,
#     "USER_TYPES": ['Entertainment', 'Home', 'Wild', 'Traffic', 'Work'],
#     "task_rates": {
#         'Entertainment': 160,
#         'Home': 60,
#         'Wild': 40,
#         'Traffic': 120,
#         'Work': 240
#     },
#     "task_prob": {
#         'Entertainment': 0.6,
#         'Home': 0.5,
#         'Wild': 0.4,
#         'Traffic': 0.7,
#         'Work': 0.8
#     },
#     "steps_per_episode": 30,
# },

TEST_PARAMS_LIST = [
    {  # environment config 1
        "NUM_USERS": 15,
        "USER_TYPES": ['Entertainment', 'Home', 'Wild', 'Traffic', 'Work'],
        "task_rates": {
            'Entertainment': 100,
            'Home': 30,
            'Wild': 20,
            'Traffic': 200,
            'Work': 120
        },
        "task_prob": {
            'Entertainment': 0.6,
            'Home': 0.5,
            'Wild': 0.4,
            'Traffic': 0.7,
            'Work': 0.8
        },
        "steps_per_episode": 30,
    },
    {  # environment config 5 (= config 1)
        "NUM_USERS": 15,
        "USER_TYPES": ['Entertainment', 'Home', 'Wild', 'Traffic', 'Work'],
        "task_rates": {
            'Entertainment': 80,
            'Home': 20,
            'Wild': 15,
            'Traffic': 160,
            'Work': 100
        },
        "task_prob": {
            'Entertainment': 0.6,
            'Home': 0.5,
            'Wild': 0.4,
            'Traffic': 0.7,
            'Work': 0.8
        },
        "steps_per_episode": 30,
    },
    {  # environment config 2 (example, adjust as needed)
        "NUM_USERS": 15,
        "USER_TYPES": ['Entertainment', 'Home', 'Wild', 'Traffic', 'Work'],
        "task_rates": {
            'Entertainment': 120,
            'Home': 80,
            'Wild': 40,
            'Traffic': 160,
            'Work': 140
        },
        "task_prob": {
            'Entertainment': 0.5,
            'Home': 0.4,
            'Wild': 0.3,
            'Traffic': 0.6,
            'Work': 0.7
        },
        "steps_per_episode": 30,
    },
    {  # environment config 3 (example, adjust as needed)
        "NUM_USERS": 15,
        "USER_TYPES": ['Entertainment', 'Home', 'Wild', 'Traffic', 'Work'],
        "task_rates": {
            'Entertainment': 80,
            'Home': 40,
            'Wild': 30,
            'Traffic': 120,
            'Work': 150
        },
        "task_prob": {
            'Entertainment': 0.4,
            'Home': 0.3,
            'Wild': 0.2,
            'Traffic': 0.5,
            'Work': 0.6
        },
        "steps_per_episode": 30,
    },
    # {  # environment config 4 (example, adjust as needed)
    #     "NUM_USERS": 25,
    #     "USER_TYPES": ['Entertainment', 'Home', 'Wild', 'Traffic', 'Work'],
    #     "task_rates": {
    #         'Entertainment': 150,
    #         'Home': 90,
    #         'Wild': 50,
    #         'Traffic': 180,
    #         'Work': 220
    #     },
    #     "task_prob": {
    #         'Entertainment': 0.7,
    #         'Home': 0.6,
    #         'Wild': 0.4,
    #         'Traffic': 0.5,
    #         'Work': 0.3
    #     },
    #     "steps_per_episode": 30,
    # },
    # {  # environment config 5 (= config 1)
    #     "NUM_USERS": 15,
    #     "USER_TYPES": ['Entertainment', 'Home', 'Wild', 'Traffic', 'Work'],
    #     "task_rates": {
    #         'Entertainment': 140,
    #         'Home': 30,
    #         'Wild': 50,
    #         'Traffic': 180,
    #         'Work': 240
    #     },
    #     "task_prob": {
    #         'Entertainment': 0.6,
    #         'Home': 0.5,
    #         'Wild': 0.4,
    #         'Traffic': 0.7,
    #         'Work': 0.8
    #     },
    #     "steps_per_episode": 30,
    # },
]

# Test-phase environment switching period (episodes), shared by DC-ELLA, the
# local learner and the transfer baseline (Fig. 8: a new environment every 25 episodes).
ENV_SWITCH_EPISODES = 25

# Note: indices of the resolution / latency requirement arrays map one-to-one to the environments (Entertainment, Home, Wild, Traffic, Work)
resolution_requirement_user = [2048*1080, 3072*1620, 4096*2048, 1280*720, 1920*1080]
time_requirement_user       = [2, 2.5, 3, 1.5, 1]

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Server-side hit and similarity discounts (for statistics)
server_hit_discount_list = [[] for _ in range(NUM_UAVS)]
server_sim_discount_list = [[] for _ in range(NUM_UAVS)]

##################################################
# ========== Initialization: user / server helpers =========
##################################################
def initialize_users():
    positions = np.random.rand(NUM_USERS, 2) * MAPSIZE
    env_types = ['Entertainment', 'Home', 'Wild', 'Traffic', 'Work']
    base = NUM_USERS // len(env_types)
    remainder = NUM_USERS % len(env_types)
    users = []
    user_index = 0
    for e_idx, env in enumerate(env_types):
        count = base + (1 if e_idx < remainder else 0)
        for _ in range(count):
            user = {
                'id': user_index + 1,
                'position': (*positions[user_index], 0),
                'environment': env,
                'environment_index': e_idx,
                'requests': [],
                'received': []
            }
            users.append(user)
            user_index += 1
    print(f"[INIT] Initialize {len(users)} users.")
    return users


##################################################
# ========== compute_gae + run flow ==========
##################################################
def user_compute_gae_single(buffer, gamma=0.99, lam=0.95):
    T = len(buffer["rewards"])
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        r_t = buffer["rewards"][t]
        v_t = buffer["values"][t, 0]
        if t == T-1:
            next_non_terminal = 0.0 if buffer["dones"][t] else 1.0
            next_value = 0.0 if buffer["dones"][t] else v_t
        else:
            next_non_terminal = 0.0 if buffer["dones"][t+1] else 1.0
            next_value = buffer["values"][t+1, 0] if not buffer["dones"][t+1] else 0.0
        delta = r_t + gamma * next_value * next_non_terminal - v_t
        adv[t] = last_gae = delta + gamma * lam * next_non_terminal * last_gae
    buffer["adv"] = adv
    buffer["returns"] = (adv + buffer["values"].squeeze(-1)) * 1e-4

def server_compute_gae_single(buffer, gamma=0.99, lam=0.95):
    T = len(buffer["rewards"])
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        r_t = buffer["rewards"][t]
        v_t = buffer["values"][t, 0]
        if t == T-1:
            next_non_terminal = 0.0 if buffer["dones"][t] else 1.0
            next_value = 0.0 if buffer["dones"][t] else v_t
        else:
            next_non_terminal = 0.0 if buffer["dones"][t+1] else 1.0
            next_value = buffer["values"][t+1, 0] if not buffer["dones"][t+1] else 0.0
        delta = r_t + gamma * next_value * next_non_terminal - v_t
        adv[t] = last_gae = delta + gamma * lam * next_non_terminal * last_gae
    buffer["adv"] = adv
    buffer["returns"] = (adv + buffer["values"].squeeze(-1)) * 1e-7  #1e-6

# run_mappo takes an enable_similarity flag and passes it to the environment constructor;
# when server caching is disabled, server_agent is not built and server rewards/training default to 0
def collect_episode_data(env, server_agent, steps_per_episode, version):
    """
    Run the environment for one episode, collect per-step buffers and statistics, grouped by user environment and server category.
    Returns: user_buffers, server_buffers, episode_stats, extra_stats
    """
    # Initialize buffers
    user_buffers = [{"obs": [], "actions": [], "logp": [], "values": [], "rewards": [], "dones": []}
                    for _ in range(env.num_agents)]
    server_buffers = [
        {"obs": [], "actions": [], "logp": [], "values": [], "rewards": [], "partial_rews": [], "dones": []}
        for _ in range(NUM_UAVS)]

    # Initialize accumulated statistics
    sum_local_user = 0.0; count_local_user = 0
    sum_global_user = 0.0; count_global_user = 0
    sum_comb_user = 0.0;   count_comb_user = 0
    sum_local_server = 0.0; count_local_server = 0
    sum_global_server = 0.0; count_global_svr = 0
    sum_comb_server = 0.0;   count_comb_svr = 0

    # Statistics per user environment
    user_types = ['Entertainment', 'Home', 'Wild', 'Traffic', 'Work']
    user_local_by_env_all = {env_type: [] for env_type in user_types}
    user_combined_by_env_all = {env_type: [] for env_type in user_types}

    # Statistics per server category
    server_categories = [1, 2, 3, 4, 5]
    server_local_by_cat_all = {cat: [] for cat in server_categories}
    server_combined_by_cat_all = {cat: [] for cat in server_categories}
    server_hit_by_uav_all = {j: [] for j in range(NUM_UAVS)}
    server_sim_by_uav_all = {j: [] for j in range(NUM_UAVS)}
    server_hit_by_cat_all = {cat: [] for cat in server_categories}
    server_queue_by_cat_all = {cat: [] for cat in server_categories}


    # Record each server's queue length and discount data (every step)
    uav_queue_history = [[] for _ in range(NUM_UAVS)]
    server_hit_discount = [[] for _ in range(NUM_UAVS)]
    server_sim_discount = [[] for _ in range(NUM_UAVS)]
    # Container for collecting server features
    server_features = {j: [] for j in range(NUM_UAVS)}

    steps = 0
    while env.current_step_in_episode < env.episode_length:
        # obs_next, done, info_all, server_feature_list = env.step(server_agent, version)
        obs_next, done, info_all, server_feature_list = env.step(server_agent)
        # User-side data
        info_user = info_all["info_user"]
        user_locals = info_user["local_r"]
        user_comb = info_user["combined_r"]
        user_acts = info_user["actions"]
        user_logps = info_user["logps"]
        user_vals = info_user["values"]
        # if any(abs(r) > 1e-9 for r in user_locals):
        sum_local_user = 0
        count_local_user += 1
    # if any(abs(c) > 1e-9 for c in user_comb):
        sum_comb_user = 0
        count_comb_user += 1
        g_r = float(info_all["global_r"])
        if abs(g_r) > 1e-9:
            sum_global_user += g_r
            count_global_user += 1
        for i in range(env.num_agents):
            if user_acts[i] == -1:
                continue
            user_buffers[i]["obs"].append(obs_next[i].reshape(1, env.obs_dim_per_agent))
            user_buffers[i]["actions"].append(user_acts[i])
            user_buffers[i]["logp"].append(user_logps[i])
            user_buffers[i]["values"].append(user_vals[i])
            user_buffers[i]["rewards"].append(0)
            user_buffers[i]["dones"].append(done)
            # Group by user environment
            env_type = env.users[i]['environment']
            user_local_by_env_all[env_type].append(user_locals[i])
            user_combined_by_env_all[env_type].append(0)

        # Server-side data
        info_server = info_all["info_server"]
        serv_locals = info_server.get("local_r", [])
        serv_global = info_server.get("global_r", 0.0)
        serv_comb = info_server.get("combined_r", [])
        serv_acts = info_server.get("actions", [])
        serv_logps = info_server.get("logps", [])
        serv_vals = info_server.get("values", [])
        partial_dict = info_server.get("partial_rews", {})
        if serv_locals and any(abs(r) > 1e-9 for r in serv_locals):
            sum_local_server += float(np.mean([r for r in serv_locals if abs(r) > 1e-9]))
            count_local_server += 1
        if abs(serv_global) > 1e-9:
            sum_global_server += serv_global
            count_global_svr += 1
        if serv_comb and any(abs(c) > 1e-9 for c in serv_comb):
            sum_comb_server += float(np.mean([c for c in serv_comb if abs(c) > 1e-9]))
            count_comb_svr += 1
        if env.enable_server_caching and server_agent is not None:
            for j in range(NUM_UAVS):
                if abs(serv_locals[j]) < 1e-9:
                    continue
                obs_j = info_server["obs_list"][j]
                act_j = serv_acts[j]
                lp_j = serv_logps[j]
                v_j = serv_vals[j]

                # Record statistics per server category
                server_cat = env.uavs[j]['category']  # env.uavs[j]['category'] must be defined
                server_local_by_cat_all[server_cat].append(serv_locals[j])
                server_combined_by_cat_all[server_cat].append(serv_comb[j])

                # Record hit and similarity (reuse) discounts
                latest_stats = env.uavs[j]['stats_history'][-1] if env.uavs[j]['stats_history'] else {}
                hit_disc = latest_stats.get('hit_discount', 0.0)
                sim_disc = latest_stats.get('sim_discount', 0.0)

                server_hit_by_uav_all[j].append(hit_disc)
                server_sim_by_uav_all[j].append(sim_disc)
                server_hit_by_cat_all[server_cat].append(hit_disc)

                # Record server queue lengths
                queue_len = len(env.uavs[j]['queue']) + len(env.uavs[j]['downlink_queue'])
                server_queue_by_cat_all[server_cat].append(queue_len)


                # Call forward to obtain the feature
                logits, _, feature = server_agent.actors[j].forward(
                    torch.tensor(obs_j.reshape(1, -1), dtype=torch.float32).to(
                        server_agent.actors[j].fc_out.weight.device)
                )

                # # Save the feature (CPU numpy array)
                # server_features[j].append(feature.detach().cpu().numpy())

                # Save the feature (from server_feature_list[j])
                server_features[j].append(server_feature_list[j])

                server_buffers[j]["obs"].append(obs_j.reshape(1, server_agent.obs_dim))
                server_buffers[j]["actions"].append(act_j)
                server_buffers[j]["logp"].append(lp_j)
                server_buffers[j]["values"].append(v_j)
                server_buffers[j]["rewards"].append(serv_comb[j])
                server_buffers[j]["dones"].append(done)
                if j in partial_dict:
                    server_buffers[j]["partial_rews"].append(partial_dict[j])
                else:
                    server_buffers[j]["partial_rews"].append(np.zeros((server_agent.action_dim,), dtype=np.float32))
        # Record server queue and discount data
        for j in range(NUM_UAVS):
            q_len = len(env.uavs[j]['queue']) + len(env.uavs[j]['downlink_queue'])
            uav_queue_history[j].append(q_len)
            if env.uavs[j]['stats_history']:
                latest_stats = env.uavs[j]['stats_history'][-1]
                hit_disc = latest_stats.get('hit_discount', 0.0)
                sim_disc = latest_stats.get('sim_discount', 0.0)
            else:
                hit_disc = sim_disc = 0.0
            server_hit_discount[j].append(hit_disc)
            server_sim_discount[j].append(sim_disc)
        steps += 1

    # Collect non-empty values before computing overall_discount
    all_discount = [env.uavs[j]['stats_history'][-1].get('overall_discount', 0.0)
                    for j in range(NUM_UAVS) if env.uavs[j]['stats_history']]
    overall_discount = np.mean(all_discount) if len(all_discount) > 0 else 0.0

    episode_stats = {
        "user_avg_local": sum_local_user / count_local_user if count_local_user > 0 else 0.0,
        "user_avg_global": sum_global_user / count_global_user if count_global_user > 0 else 0.0,
        "user_avg_combined": sum_comb_user / count_comb_user if count_comb_user > 0 else 0.0,
        "server_avg_local": sum_local_server / count_local_server if count_local_server > 0 else 0.0,
        "server_avg_global": sum_global_server / count_global_svr if count_global_svr > 0 else 0.0,
        "server_avg_combined": sum_comb_server / count_comb_svr if count_comb_svr > 0 else 0.0,
        "overall_discount": overall_discount,
    }
    # Assemble extra statistics
    extra_stats = {
        "user_local_by_env": {env: np.mean(vals) if vals else 0.0 for env, vals in user_local_by_env_all.items()},
        "user_combined_by_env": {env: np.mean(vals) if vals else 0.0 for env, vals in user_combined_by_env_all.items()},
        "server_local_by_cat": {cat: np.mean(vals) if vals else 0.0 for cat, vals in server_local_by_cat_all.items()},
        "server_combined_by_cat": {cat: np.mean(vals) if vals else 0.0 for cat, vals in server_combined_by_cat_all.items()},
        "server_hit_by_uav": {j: np.mean(vals) if vals else 0.0 for j, vals in server_hit_by_uav_all.items()},
        "server_sim_by_uav": {j: np.mean(vals) if vals else 0.0 for j, vals in server_sim_by_uav_all.items()},
        "server_hit_by_cat": {cat: np.mean(vals) if vals else 0.0 for cat, vals in server_hit_by_cat_all.items()},
        "server_queue_by_cat": {cat: np.mean(vals) if vals else 0.0 for cat, vals in server_queue_by_cat_all.items()},
        "uav_queue_history": uav_queue_history,
        "server_hit_discount": server_hit_discount,
        "server_sim_discount": server_sim_discount,
    }

    # At the end of the episode, compress the features into one vector (task descriptor Phi)
    server_feature_avg = {}
    for j in range(NUM_UAVS):
        if len(server_features[j]) > 0:
            server_feature_avg[j] = np.mean(np.vstack(server_features[j]), axis=0).flatten()
        else:
            if server_agent is not None:
                feature_size = 202
            else:
                feature_size = 202  # task-feature dimension (202 in the paper)
            server_feature_avg[j] = np.zeros((feature_size,))
        print('server =', j, 'feature_vec =', server_feature_avg)

    # Also return server_feature_avg
    return user_buffers, server_buffers, episode_stats, extra_stats, server_feature_avg


def update_agents(env, server_buffers, server_agent):
    total_actor_loss, total_critic_loss, total_loss = 0.0, 0.0, 0.0

    user_loss = 0

    # Policy-parameter dimension d (taken from server_agent)
    if server_agent:
        dummy_theta = server_agent.actors[0].extract_policy_parameters()
        d = dummy_theta.shape[0]
    else:
        d = 1  # avoids errors when there is no server_agent

    server_hess = [np.eye(d) for _ in range(NUM_UAVS)]

    # # User update logic (unchanged)
    # for i in range(env.num_agents):
    #     buf = user_buffers[i]
    #     T = len(buf["rewards"])
    #     if T == 0:
    #         continue
    #     obs_arr = np.zeros((T, 1, env.obs_dim_per_agent), dtype=np.float32)
    #     for t in range(T):
    #         obs_arr[t] = buf["obs"][t]
    #     buf["obs"] = obs_arr
    #     buf["actions"] = np.array(buf["actions"], dtype=np.int32).reshape(T, 1)
    #     buf["logp"] = np.array(buf["logp"], dtype=np.float32).reshape(T, 1)
    #     buf["values"] = np.array(buf["values"], dtype=np.float32).reshape(T, 1)
    #     buf["rewards"] = np.array(buf["rewards"], dtype=np.float32)
    #     buf["dones"] = np.array(buf["dones"], dtype=bool)
    #     user_compute_gae_single(buf, gamma=user_agent.gamma, lam=user_agent.lam)
    #     a_l, c_l, tot_l = user_agent.update_single(i, buf, train_iters=10, batch_size=64)
    #     total_actor_loss += a_l
    #     total_critic_loss += c_l
    #     total_loss += tot_l
    # user_loss = (total_actor_loss / env.num_agents,
    #              total_critic_loss / env.num_agents,
    #              total_loss / env.num_agents)

    serv_actor_loss, serv_critic_loss, serv_tot_loss, valid_server_count = 0.0, 0.0, 0.0, 0
    if server_agent:
        for j in range(NUM_UAVS):
            buf = server_buffers[j]
            T = len(buf["rewards"])
            if T == 0:
                continue
            valid_server_count += 1
            obs_arr = np.zeros((T, 1, server_agent.obs_dim), dtype=np.float32)
            partial_arr = [buf["partial_rews"][t] for t in range(T)]
            buf["obs"] = obs_arr
            buf["actions"] = np.array(buf["actions"], dtype=np.int32)
            buf["logp"] = np.array(buf["logp"], dtype=np.float32).reshape(T)
            buf["values"] = np.array(buf["values"], dtype=np.float32).reshape(T, 1)
            buf["rewards"] = np.array(buf["rewards"], dtype=np.float32)
            buf["dones"] = np.array(buf["dones"], dtype=bool)
            buf["partial_rews"] = np.stack(partial_arr, axis=0)
            server_compute_gae_single(buf, gamma=server_agent.gamma, lam=server_agent.lam)

            # update_single returns 4 values (actor_loss, critic_loss, total_loss, local_hess)
            a_l, c_l, tot_l, local_hess = server_agent.update_single(j, buf, train_iters=20, batch_size=128)
            serv_actor_loss += a_l
            serv_critic_loss += c_l
            serv_tot_loss += tot_l

            # Store the computed Hessian
            server_hess[j] = local_hess

    server_loss = (
        serv_actor_loss / valid_server_count if valid_server_count else 0.0,
        serv_critic_loss / valid_server_count if valid_server_count else 0.0,
        serv_tot_loss / valid_server_count if valid_server_count else 0.0
    )

    # Return the list of Hessian matrices
    return user_loss, server_loss, server_hess


def run_mappo_pretrain(enable_server_caching=True, enable_similarity=True, case_label="default"):
    print(f"[RUN] Pretrain (server_caching={enable_server_caching}, similarity={enable_similarity})")
    print(f"[RUN] Using PRETRAIN environment with reduced complexity: \n{PRETRAIN_PARAMS}")
    # Pretraining environment parameters
    pretrain_steps = PRETRAIN_PARAMS["steps_per_episode"]
    num_episodes, steps_per_episode = 25, 30

    version = 1

    server_agent = ServerMAPPOAgent(
        num_servers=NUM_UAVS, obs_dim=200, action_dim=100, hidden_dim=32,
        lr=1e-4, gamma=0.99, lam=0.95, alpha=0.5, beta=0.5, device=device,
        clip_ratio=0.4, entropy_coef=0.1, max_grad_norm=0.5
    ) if enable_server_caching else None

    # Initialize the DC-ELLA learner
    dim_theta = server_agent.actors[0].extract_policy_parameters().shape[0]

    ella_learner = ServerLifelongLearner(
        dim_theta=dim_theta,
        dim_feature=202,  # task-feature dimension
        dim_latent=8,
        mu_one=1e-5,
        mu_two=1e-5,
        lr=1e-3
    )

    # Statistics
    user_avg_local_list, user_avg_global_list, user_avg_combined_list = [], [], []
    server_avg_local_list, server_avg_global_list, server_avg_combined_list = [], [], []
    server_discount_list, user_critic_loss_list, server_critic_loss_list, server_hit_rate_list = [], [], [], []
    task_count_over_episodes = []

    # (1) Log task_id -> [(episode, server_j), ...]
    task_occurrence_log = defaultdict(list)
    # (2) Log policy-update events: each element is a dict
    policy_change_events = []  # each entry: {"episode": ep, "server": j,
    #        "prev": old_id, "new": new_id}

    # Use the user types of the pretraining environment rather than the full set
    user_types = PRETRAIN_PARAMS["USER_TYPES"]
    server_categories = [1, 2, 3, 4, 5]

    user_local_by_env_all = {env: [] for env in user_types}
    user_combined_by_env_all = {env: [] for env in user_types}
    server_local_by_cat_all = {cat: [] for cat in server_categories}
    server_combined_by_cat_all = {cat: [] for cat in server_categories}
    server_hit_by_uav_all, server_sim_by_uav_all = {j: [] for j in range(NUM_UAVS)}, {j: [] for j in range(NUM_UAVS)}
    server_hit_by_cat_all, server_queue_by_cat_all = {cat: [] for cat in server_categories}, {cat: [] for cat in server_categories}
    uav_queue_history_all, server_feature_history = [], {j: [] for j in range(NUM_UAVS)}

    uav_categories = None

    # Dict holding each server's previous task id
    previous_task_ids = {j: None for j in range(NUM_UAVS)}

    for ep in range(num_episodes):

        # Create the environment with the pretraining configuration
        env = MultiUserEnv(
            episode_length=pretrain_steps, time_slot=1.0, alpha=1, beta=0,
            NUM_USERS = NUM_USERS, NUM_UAVS = NUM_UAVS, task_rates = task_rates, task_prob = task_prob, TEST_PARAMS = PRETRAIN_PARAMS,
            MAPSIZE = MAPSIZE, resolution_requirement_user = resolution_requirement_user, time_requirement_user = time_requirement_user, TASK_SIZE = TASK_SIZE, UAV_HEIGHT = UAV_HEIGHT,
            subject_similarity_matrix = subject_similarity_matrix, object_similarity_matrix = object_similarity_matrix, BIT_PER_CYCLE = BIT_PER_CYCLE,
            mu_compute=mu_compute, mu_downlink=mu_downlink, COMPRESSION_RATE=COMPRESSION_RATE,
            channel_params=channel_params, bandwidth_map_user=bandwidth_map_user, CARRIER_FREQ=CARRIER_FREQ,
            P_USER=P_USER, G_USER=G_USER, NOISE_POWER=NOISE_POWER, P_UAV=P_UAV, G_UAV=G_UAV, image_label_mapping=image_label_mapping,
            enable_server_caching=enable_server_caching,
            enable_similarity=enable_similarity,
            env_params=PRETRAIN_PARAMS  # pretraining environment parameters
        )

        print(f"[RUN] Pretrain Episode {ep}/{num_episodes}")
        obs_list = env.reset()
        if uav_categories is None:
            uav_categories = [uav['category'] for uav in env.uavs]

        # user_agent.reset_hidden_states()

        if server_agent:
            server_agent.reset_hidden_states()

        user_buffers, server_buffers, ep_stats, extra_stats, server_feature_avg = collect_episode_data(
            env, server_agent, steps_per_episode, version
        )

        # Store each server's mean feature
        for j in range(NUM_UAVS):
            server_feature_history[j].append(server_feature_avg[j])

        # Update statistics
        user_avg_local_list.append(ep_stats["user_avg_local"])
        user_avg_global_list.append(ep_stats["user_avg_global"])
        user_avg_combined_list.append(ep_stats["user_avg_combined"])
        server_avg_local_list.append(ep_stats["server_avg_local"])
        server_avg_global_list.append(ep_stats["server_avg_global"])
        server_avg_combined_list.append(ep_stats["server_avg_combined"])
        server_discount_list.append(ep_stats["overall_discount"])
        uav_queue_history_all.append(np.mean(env.server_queue_history[-1]))

        # Global server hit rate
        server_hit_rate_ep = np.mean([
            uav['stats_history'][-1].get('hit_rate', 0.0) if uav['stats_history'] else 0.0
            for uav in env.uavs
        ])
        server_hit_rate_list.append(server_hit_rate_ep)

        # for env_type in user_types:
        #     user_local_by_env_all[env_type].append(extra_stats["user_local_by_env"][env_type])
        #     user_combined_by_env_all[env_type].append(extra_stats["user_combined_by_env"][env_type])
        # Only collect statistics for user types that exist in the pretraining environment
        for env_type in user_types:
            if env_type in extra_stats["user_local_by_env"]:
                user_local_by_env_all[env_type].append(extra_stats["user_local_by_env"][env_type])
            if env_type in extra_stats["user_combined_by_env"]:
                user_combined_by_env_all[env_type].append(extra_stats["user_combined_by_env"][env_type])

        for j, uav in enumerate(env.uavs):
            cat = uav['category']
            latest_stats = uav['stats_history'][-1] if uav['stats_history'] else {}
            server_hit_by_uav_all[j].append(latest_stats.get('hit_discount', 0.0))
            server_sim_by_uav_all[j].append(latest_stats.get('sim_discount', 0.0))
            server_hit_by_cat_all[cat].append(latest_stats.get('hit_rate', 0.0))
            server_queue_by_cat_all[cat].append(len(uav['queue']) + len(uav['downlink_queue']))

        user_loss, server_loss, server_hess = update_agents(env, server_buffers, server_agent)
        user_critic_loss_list.append(0)
        server_critic_loss_list.append(server_loss[1])


        # [3] Determine the task id and compute the new s'
        server_task_list = []
        for j in range(NUM_UAVS):
            feature_j = server_feature_avg[j]
            # task_id, is_new_task = ella_learner.determine_task_id(feature_j, j)
            task_id = j
            ella_learner.init_task_if_new(task_id)

            theta_current = server_agent.actors[j].extract_policy_parameters().cpu().numpy()
            server_task_list.append((task_id, theta_current, server_hess[j], feature_j))

        # [4] Update the shared knowledge base L
        ella_learner.learn_multitask_server_actors(server_task_list)

        # [5] Update the dictionary D
        ella_learner.update_dictionary_D()

        print(f"[INFO] Episode {ep} completed.")
        print(f"  - Number of stored task features: {len(ella_learner.task_feature_dict)}")
        print(f"  - Number of stored s vectors: {len(ella_learner.dict_s)}")

        task_count_over_episodes.append(len(ella_learner.task_feature_dict))

    # Final statistics:
    final_stats = {
        "user_critic_loss": user_critic_loss_list,
        "server_critic_loss": server_critic_loss_list,
        "user_local_reward": user_avg_local_list,
        "user_global_reward": user_avg_global_list,
        "user_combined_reward": user_avg_combined_list,
        "server_local_reward": server_avg_local_list,
        "server_global_reward": server_avg_global_list,
        "server_combined_reward": server_avg_combined_list,
        "overall_discount": server_discount_list,
        "server_hit_rate": np.mean(server_hit_rate_list),
        "avg_server_queue": uav_queue_history_all[-1]
    }

    extra_stats_final = {
        "user_local_by_env": user_local_by_env_all,
        "user_combined_by_env": user_combined_by_env_all,
        "server_local_by_cat": server_local_by_cat_all,
        "server_combined_by_cat": server_combined_by_cat_all,
        "server_hit_by_uav": server_hit_by_uav_all,
        "server_sim_by_uav": server_sim_by_uav_all,
        "server_hit_by_cat": server_hit_by_cat_all,
        "server_queue_by_cat": server_queue_by_cat_all,
        "uav_queue_history": uav_queue_history_all,
        "uav_categories": uav_categories,
        # Feature history
        "server_feature_history": server_feature_history,
        # Store in the extra statistics
        "task_count_over_episodes": task_count_over_episodes,
        "task_occurrence_log": dict(task_occurrence_log),   # defaultdict -> plain dict
        "policy_change_events": policy_change_events
    }


    final_metrics = {
        "user_global_reward": final_stats["user_global_reward"][-1],
        "user_local_reward": final_stats["user_local_reward"][-1],
        "user_combined_reward": final_stats["user_combined_reward"][-1],
        "server_global_reward": final_stats["server_global_reward"][-1],
        "server_local_reward": final_stats["server_local_reward"][-1],
        "server_combined_reward": final_stats["server_combined_reward"][-1],
        "overall_discount": final_stats["overall_discount"][-1],
        "server_hit_rate": final_stats["server_hit_rate"],
        "avg_server_queue": final_stats["avg_server_queue"]
    }

    # Save the DC-ELLA learner
    save_path = "pretrained_params/ella_learner.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(ella_learner, f)
    print(f"[SAVE] ELLA Learner saved to {save_path}")

    # ----- Save actor and critic parameters -----
    for j in range(NUM_UAVS):
        actor_save_path = os.path.join("pretrained_params", f"server_actor_{j}_non_policy_params.pt")
        torch.save(server_agent.actors[j].state_dict(), actor_save_path)
        print(f"[SAVE] actor non-policy params saved to {actor_save_path}")

        critic_save_path = os.path.join("pretrained_params", f"server_critic_{j}_params.pt")
        torch.save(server_agent.critics[j].state_dict(), critic_save_path)
        print(f"[SAVE] critic params saved to {critic_save_path}")
    # -----------------------------------

    # ========== Save statistics to the task_analyse directory ==========
    base_dir   = os.path.dirname(__file__)          # directory of this .py file
    save_dir   = os.path.join(base_dir, "task_analyse")
    os.makedirs(save_dir, exist_ok=True)            # create if missing
    save_file  = os.path.join(save_dir, "pretrain_stats.pkl")
    with open(save_file, "wb") as f:
        pickle.dump(extra_stats_final, f)
    print(f"[run_mappo_pretrain] ✅ Stats saved → {save_file}")


    return final_stats, final_metrics, extra_stats_final


def run_mappo_test(enable_server_caching=True, enable_similarity=True, case_label="default", get_pretrain_params=True):
    print(f"[RUN] Test (server_caching={enable_server_caching}, similarity={enable_similarity}, pretrain={get_pretrain_params})")
    print(f"[RUN] Using FULL TEST environment with complete complexity")

    # num_episodes, steps_per_episode = 20, 30
    # Test environment parameters
    test_steps = TEST_PARAMS["steps_per_episode"]
    num_episodes = 100

    server_agent = ServerMAPPOAgent(
        num_servers=NUM_UAVS, obs_dim=200, action_dim=100, hidden_dim=32,
        lr=1e-4, gamma=0.99, lam=0.95, alpha=0.5, beta=0.5, device=device,
        clip_ratio=0.4, entropy_coef=0.1, max_grad_norm=0.5
    ) if enable_server_caching else None

    load_dir, save_dir = "./pretrained_params", "./trained_well_params"
    os.makedirs(save_dir, exist_ok=True)

    dim_theta = server_agent.actors[0].extract_policy_parameters().shape[0]

    env = MultiUserEnv(
        episode_length=test_steps, time_slot=1.0, alpha=1, beta=0,
        NUM_USERS=NUM_USERS, NUM_UAVS=NUM_UAVS, task_rates=task_rates, task_prob=task_prob, TEST_PARAMS=TEST_PARAMS,
        MAPSIZE=MAPSIZE, resolution_requirement_user=resolution_requirement_user,
        time_requirement_user=time_requirement_user, TASK_SIZE=TASK_SIZE, UAV_HEIGHT=UAV_HEIGHT,
        subject_similarity_matrix=subject_similarity_matrix, object_similarity_matrix=object_similarity_matrix, BIT_PER_CYCLE=BIT_PER_CYCLE,
        mu_compute=mu_compute, mu_downlink=mu_downlink, COMPRESSION_RATE=COMPRESSION_RATE,
        channel_params=channel_params, bandwidth_map_user=bandwidth_map_user, CARRIER_FREQ=CARRIER_FREQ,
        P_USER=P_USER, G_USER=G_USER, NOISE_POWER=NOISE_POWER, P_UAV=P_UAV, G_UAV=G_UAV, image_label_mapping=image_label_mapping,
        enable_server_caching=enable_server_caching,
        enable_similarity=enable_similarity,
        env_params=TEST_PARAMS  # test environment parameters
    )

    obs_list = env.reset()

    if get_pretrain_params:
        for j in range(NUM_UAVS):
            actor_path = os.path.join(load_dir, f"server_actor_{j}_non_policy_params.pt")
            server_agent.actors[j].load_state_dict(torch.load(actor_path), strict=False)
            # server_agent.actors[j].load_state_dict(torch.load(actor_path, map_location=torch.device('cpu')),
            #                                        strict=False)
            print(f"[LOAD] Loaded non-policy params for actor {j}")


            critic_path = os.path.join(load_dir, f"server_critic_{j}_params.pt")
            server_agent.critics[j].load_state_dict(torch.load(critic_path), strict=False)
            # server_agent.critics[j].load_state_dict(torch.load(critic_path, map_location=torch.device('cpu')),
            #                                         strict=False)
            print(f"[LOAD] Loaded params for critic {j}")

        ella_learner_path = os.path.join(load_dir, "ella_learner.pkl")
        with open(ella_learner_path, "rb") as f:
            ella_learner = pickle.load(f)
        print(f"[LOAD] ELLA learner loaded from {ella_learner_path}")


        # # Get the initial environment feature
        # # user_agent.reset_hidden_states()
        if server_agent:
            server_agent.reset_hidden_states()

        _, _, _, _, initial_server_feature_avg = collect_episode_data(
            env, server_agent, steps_per_episode=30, version=1
        )

        # First solve s explicitly from the environment feature and dictionary D (Eq. 27)
        for j in range(NUM_UAVS):
            task_id = j + NUM_UAVS
            feature_test = initial_server_feature_avg[j]
            ella_learner.init_task_if_new(task_id)
            ella_learner.compute_s_from_feature(feature_test, task_id)

        # Push the policy parameters rebuilt from the updated L and s back to the servers
        for j in range(NUM_UAVS):
            task_id = j + NUM_UAVS
            theta_j = ella_learner.L @ ella_learner.dict_s[task_id]
            server_agent.actors[j].load_policy_parameters(torch.from_numpy(theta_j).float())
            print(f"[INIT TEST] Loaded initial policy params for actor {j}, task_id={task_id}")

    else:
        ella_learner = ServerLifelongLearner(
            dim_theta=dim_theta,
            dim_feature=202,  # task-feature dimension
            dim_latent=8,
            mu_one=1e-5,
            mu_two=1e-5,
            lr=1e-3
        )

    print("[CREATE] New ELLA learner created.")

    user_avg_local_list, user_avg_global_list, user_avg_combined_list = [], [], []
    server_avg_local_list, server_avg_global_list, server_avg_combined_list = [], [], []
    server_discount_list, user_critic_loss_list, server_critic_loss_list, server_hit_rate_list = [], [], [], []
    task_count_over_episodes = []  # defined before the episode loop


    # (1) Log task_id -> [(episode, server_j), ...]
    task_occurrence_log = defaultdict(list)
    # (2) Log policy-update events: each element is a dict
    policy_change_events = []  # each entry: {"episode": ep, "server": j,
    #        "prev": old_id, "new": new_id}


    # user_types = ['Entertainment', 'Home', 'Wild', 'Traffic', 'Work']
    # Use the user types of the test environment rather than the pretraining one
    user_types = TEST_PARAMS["USER_TYPES"]
    server_categories = [1, 2, 3, 4, 5]

    # Extra statistics containers (accumulated over all episodes)
    user_local_by_env_all = {env: [] for env in user_types}
    user_combined_by_env_all = {env: [] for env in user_types}
    server_local_by_cat_all = {cat: [] for cat in server_categories}
    server_combined_by_cat_all = {cat: [] for cat in server_categories}
    server_hit_by_uav_all = {j: [] for j in range(NUM_UAVS)}
    server_sim_by_uav_all = {j: [] for j in range(NUM_UAVS)}
    server_hit_by_cat_all = {cat: [] for cat in server_categories}
    server_queue_by_cat_all = {cat: [] for cat in server_categories}
    server_feature_history = {j: [] for j in range(NUM_UAVS)}
    uav_queue_history_all, uav_categories = [], None

    # Dict holding each server's previous task id
    previous_task_ids = {j: None for j in range(NUM_UAVS)}

    for ep in range(num_episodes):

        print(f"[RUN] Episode {ep}/{num_episodes}")

        # Version label for each environment update (must be inside the loop)
        version = ep // ENV_SWITCH_EPISODES + 1
        prev_version = version - 1
        version_for_task = version + 1

        # Switch the environment parameters every ENV_SWITCH_EPISODES episodes
        current_test_params = TEST_PARAMS_LIST[(ep // ENV_SWITCH_EPISODES) % len(TEST_PARAMS_LIST)]

        env = MultiUserEnv(
            episode_length=current_test_params["steps_per_episode"], time_slot=1.0, alpha=1, beta=0,
            NUM_USERS=NUM_USERS, NUM_UAVS=NUM_UAVS, task_rates=task_rates, task_prob=task_prob, TEST_PARAMS=TEST_PARAMS,
            MAPSIZE=MAPSIZE, resolution_requirement_user=resolution_requirement_user,
            time_requirement_user=time_requirement_user, TASK_SIZE=TASK_SIZE, UAV_HEIGHT=UAV_HEIGHT,
            subject_similarity_matrix=subject_similarity_matrix, object_similarity_matrix=object_similarity_matrix, BIT_PER_CYCLE=BIT_PER_CYCLE,
            mu_compute=mu_compute, mu_downlink=mu_downlink, COMPRESSION_RATE=COMPRESSION_RATE,
            channel_params=channel_params, bandwidth_map_user=bandwidth_map_user, CARRIER_FREQ=CARRIER_FREQ,
            P_USER=P_USER, G_USER=G_USER, NOISE_POWER=NOISE_POWER, P_UAV=P_UAV, G_UAV=G_UAV, image_label_mapping=image_label_mapping,
            enable_server_caching=enable_server_caching,
            enable_similarity=enable_similarity,
            env_params=current_test_params  # current test environment parameters
        )

        print(f"[ENV RE-CREATED] Episode {ep}, new params applied.")

        obs_list = env.reset()

        # Keep user_types and steps_per_episode in sync
        user_types = current_test_params["USER_TYPES"]
        test_steps = current_test_params["steps_per_episode"]

        if uav_categories is None:
            uav_categories = [uav['category'] for uav in env.uavs]

        # user_agent.reset_hidden_states()
        if server_agent:
            server_agent.reset_hidden_states()

        print(f"[Start Normal Working] ep={ep}")
        # Before the actual test-phase collection starts:
        obs = env.reset()  # reset again


        user_buffers, server_buffers, ep_stats, extra_stats, server_feature_avg = collect_episode_data(
            env, server_agent, test_steps, version
        )


        # Store each server's mean feature
        for j in range(NUM_UAVS):
            server_feature_history[j].append(server_feature_avg[j])

        # Accumulate per-user-environment and per-server-category statistics over all episodes
        for env_type in user_types:
            user_local_by_env_all[env_type].append(extra_stats["user_local_by_env"].get(env_type, 0.0))
            user_combined_by_env_all[env_type].append(extra_stats["user_combined_by_env"].get(env_type, 0.0))

        for cat in server_categories:
            server_local_by_cat_all[cat].append(extra_stats["server_local_by_cat"].get(cat, 0.0))
            server_combined_by_cat_all[cat].append(extra_stats["server_combined_by_cat"].get(cat, 0.0))
            server_hit_by_cat_all[cat].append(extra_stats["server_hit_by_cat"].get(cat, 0.0))
            server_queue_by_cat_all[cat].append(extra_stats["server_queue_by_cat"].get(cat, 0.0))

        for j in range(NUM_UAVS):
            server_hit_by_uav_all[j].append(extra_stats["server_hit_by_uav"].get(j, 0.0))
            server_sim_by_uav_all[j].append(extra_stats["server_sim_by_uav"].get(j, 0.0))

        user_avg_local_list.append(ep_stats["user_avg_local"])
        user_avg_global_list.append(ep_stats["user_avg_global"])
        user_avg_combined_list.append(ep_stats["user_avg_combined"])
        server_avg_local_list.append(ep_stats["server_avg_local"])
        server_avg_global_list.append(ep_stats["server_avg_global"])
        server_avg_combined_list.append(ep_stats["server_avg_combined"])
        server_discount_list.append(ep_stats["overall_discount"])
        uav_queue_history_all.append(np.mean(env.server_queue_history[-1]))

        server_hit_rate_ep = np.mean(
            [uav['stats_history'][-1].get('hit_rate', 0.0) if uav['stats_history'] else 0.0 for uav in env.uavs])
        server_hit_rate_list.append(server_hit_rate_ep)

        user_loss, server_loss, server_hess = update_agents(env, server_buffers,
                                                            server_agent)
        user_critic_loss_list.append(0)
        server_critic_loss_list.append(server_loss[1])

        # Record the current number of tasks at the end of the episode
        task_count_over_episodes.append(0)

        # [3] Determine the task id and compute the new s' (DC-ELLA update and task detection)
        # task_ids = [None] * NUM_UAVS           # task_id of every server in this round
        # Explicit procedure at every environment switch (including ep == 0)
        if ep % ENV_SWITCH_EPISODES == 0:
            # so that learn_multitask_server_actors does not recompute s afterwards
            server_task_list = []
            for j in range(NUM_UAVS):
                task_id = j + version * NUM_UAVS
                theta_current = server_agent.actors[j].extract_policy_parameters().detach().cpu().numpy()
                server_task_list.append((task_id, theta_current, server_hess[j], server_feature_avg[j]))

            ella_learner.learn_multitask_server_actors(server_task_list, recompute_s=True)

            # Push the policy parameters rebuilt from the updated L and s back to the servers
            for j in range(NUM_UAVS):
                task_id = j + version * NUM_UAVS
                theta_j = ella_learner.L @ ella_learner.dict_s[task_id]
                server_agent.actors[j].load_policy_parameters(torch.from_numpy(theta_j).float())
                print(
                    f"[GUIDANCE-2] ep={ep}: version {prev_version}->{version}, "
                    f"loaded policy for actor {j}, task_id={task_id}"
                )

        # Not an environment-switch episode: regular update
        else:
            server_task_list = []
            for j in range(NUM_UAVS):
                task_id = j + version * NUM_UAVS
                feature_test = server_feature_avg[j]
                theta_current = server_agent.actors[j].extract_policy_parameters().detach().cpu().numpy()
                server_task_list.append((task_id, theta_current, server_hess[j], feature_test))

            ella_learner.learn_multitask_server_actors(server_task_list)  # recompute_s=True by default

        ella_learner.update_dictionary_D()

        print(f"[INFO] Test Episode {ep} completed.")
        print(f"  - Number of stored task features: {len(ella_learner.task_observed)}")
        print(f"  - Number of stored s vectors: {len(ella_learner.dict_s)}")

        # Record the current number of tasks at the end of the episode
        task_count_over_episodes.append(len(ella_learner.task_observed))

    # Assemble final statistics
    final_stats = {
        "user_critic_loss": user_critic_loss_list,
        "server_critic_loss": server_critic_loss_list,
        "user_local_reward": user_avg_local_list,
        "user_global_reward": user_avg_global_list,
        "user_combined_reward": user_avg_combined_list,
        "server_local_reward": server_avg_local_list,
        "server_global_reward": server_avg_global_list,
        "server_combined_reward": server_avg_combined_list,
        "overall_discount": server_discount_list,
        "server_hit_rate": [np.mean(hr) for hr in server_hit_rate_list],
        "avg_server_queue": uav_queue_history_all[-1]
    }

    extra_stats_final = {
        "user_local_by_env": user_local_by_env_all,
        "user_combined_by_env": user_combined_by_env_all,
        "server_local_by_cat": server_local_by_cat_all,
        "server_combined_by_cat": server_combined_by_cat_all,
        "server_hit_by_uav": server_hit_by_uav_all,
        "server_sim_by_uav": server_sim_by_uav_all,
        "server_hit_by_cat": server_hit_by_cat_all,
        "server_queue_by_cat": server_queue_by_cat_all,
        "server_feature_history": server_feature_history,
        "uav_categories": uav_categories,
        # Store in the extra statistics
        "task_count_over_episodes": task_count_over_episodes,
        "task_occurrence_log": dict(task_occurrence_log),   # defaultdict -> plain dict
        "policy_change_events": policy_change_events
    }

    final_metrics = {
        "user_global_reward": final_stats["user_global_reward"][-1],
        "user_local_reward": final_stats["user_local_reward"][-1],
        "user_combined_reward": final_stats["user_combined_reward"][-1],
        "server_global_reward": final_stats["server_global_reward"][-1],
        "server_local_reward": final_stats["server_local_reward"][-1],
        "server_combined_reward": final_stats["server_combined_reward"][-1],
        "overall_discount": final_stats["overall_discount"][-1],
        "server_hit_rate": final_stats["server_hit_rate"],
        "avg_server_queue": final_stats["avg_server_queue"]
    }

    print("[RUN] All episodes done.")

    # Save the final trained DC-ELLA learner
    ella_learner_save_path = os.path.join(save_dir, "ella_learner_final.pkl")
    with open(ella_learner_save_path, "wb") as f:
        pickle.dump(ella_learner, f)
    print(f"[SAVE] ELLA learner saved to {ella_learner_save_path}")


    # ========== Save statistics to the task_analyse directory ==========
    base_dir   = os.path.dirname(__file__)
    save_dir   = os.path.join(base_dir, "task_analyse")
    os.makedirs(save_dir, exist_ok=True)
    save_file  = os.path.join(save_dir, "test_stats.pkl")
    with open(save_file, "wb") as f:
        pickle.dump(extra_stats_final, f)
    print(f"[run_mappo_test] ✅ Stats saved → {save_file}")

    return final_stats, final_metrics, extra_stats_final




def run_mappo_base_test(enable_server_caching=True, enable_similarity=True, case_label="default", get_pretrain_params=True):
    print(f"[RUN] Test (server_caching={enable_server_caching}, similarity={enable_similarity}, pretrain={get_pretrain_params})")

    num_episodes, steps_per_episode = 100, 30

    server_agent = ServerMAPPOAgent(
        num_servers=NUM_UAVS, obs_dim=200, action_dim=100, hidden_dim=32,
        lr=1e-4, gamma=0.99, lam=0.95, alpha=0.5, beta=0.5, device=device,
        clip_ratio=0.4, entropy_coef=0.1, max_grad_norm=0.5
    ) if enable_server_caching else None


    user_avg_local_list, user_avg_global_list, user_avg_combined_list = [], [], []
    server_avg_local_list, server_avg_global_list, server_avg_combined_list = [], [], []
    server_discount_list, user_critic_loss_list, server_critic_loss_list, server_hit_rate_list = [], [], [], []
    task_count_over_episodes = []

    user_types = ['Entertainment', 'Home', 'Wild', 'Traffic', 'Work']
    server_categories = [1, 2, 3, 4, 5]


    # Extra statistics containers (accumulated over all episodes)
    user_local_by_env_all = {env: [] for env in user_types}
    user_combined_by_env_all = {env: [] for env in user_types}
    server_local_by_cat_all = {cat: [] for cat in server_categories}
    server_combined_by_cat_all = {cat: [] for cat in server_categories}
    server_hit_by_uav_all = {j: [] for j in range(NUM_UAVS)}
    server_sim_by_uav_all = {j: [] for j in range(NUM_UAVS)}
    server_hit_by_cat_all = {cat: [] for cat in server_categories}
    server_queue_by_cat_all = {cat: [] for cat in server_categories}
    server_feature_history = {j: [] for j in range(NUM_UAVS)}
    uav_queue_history_all, uav_categories = [], None


    for ep in range(num_episodes):

        print(f"[RUN] Episode {ep}/{num_episodes}")

        # Version label for each environment update (must be inside the loop)
        version = ep // ENV_SWITCH_EPISODES + 1
        prev_version = version - 1
        version_for_task = version + 1

        # Switch the environment parameters every ENV_SWITCH_EPISODES episodes
        current_test_params = TEST_PARAMS_LIST[(ep // ENV_SWITCH_EPISODES) % len(TEST_PARAMS_LIST)]

        env = MultiUserEnv(
            episode_length=current_test_params["steps_per_episode"], time_slot=1.0, alpha=1, beta=0,
            NUM_USERS=NUM_USERS, NUM_UAVS=NUM_UAVS, task_rates=task_rates, task_prob=task_prob, TEST_PARAMS=TEST_PARAMS,
            MAPSIZE=MAPSIZE, resolution_requirement_user=resolution_requirement_user,
            time_requirement_user=time_requirement_user, TASK_SIZE=TASK_SIZE, UAV_HEIGHT=UAV_HEIGHT,
            subject_similarity_matrix=subject_similarity_matrix, object_similarity_matrix=object_similarity_matrix, BIT_PER_CYCLE=BIT_PER_CYCLE,
            mu_compute=mu_compute, mu_downlink=mu_downlink, COMPRESSION_RATE=COMPRESSION_RATE,
            channel_params=channel_params, bandwidth_map_user=bandwidth_map_user, CARRIER_FREQ=CARRIER_FREQ,
            P_USER=P_USER, G_USER=G_USER, NOISE_POWER=NOISE_POWER, P_UAV=P_UAV, G_UAV=G_UAV, image_label_mapping=image_label_mapping,
            enable_server_caching=enable_server_caching,
            enable_similarity=enable_similarity,
            env_params=current_test_params  # current test environment parameters
        )

        print(f"[ENV RE-CREATED] Episode {ep}, new params applied.")

        obs_list = env.reset()

        # Keep user_types and steps_per_episode in sync
        user_types = current_test_params["USER_TYPES"]
        test_steps = current_test_params["steps_per_episode"]

        if uav_categories is None:
            uav_categories = [uav['category'] for uav in env.uavs]

        # user_agent.reset_hidden_states()
        if server_agent:
            server_agent.reset_hidden_states()

        user_buffers, server_buffers, ep_stats, extra_stats, server_feature_avg = collect_episode_data(
            env, server_agent, steps_per_episode, version
        )

        # Store each server's mean feature
        for j in range(NUM_UAVS):
            server_feature_history[j].append(server_feature_avg[j])

        # Accumulate per-user-environment and per-server-category statistics over all episodes
        for env_type in user_types:
            user_local_by_env_all[env_type].append(extra_stats["user_local_by_env"].get(env_type, 0.0))
            user_combined_by_env_all[env_type].append(extra_stats["user_combined_by_env"].get(env_type, 0.0))

        for cat in server_categories:
            server_local_by_cat_all[cat].append(extra_stats["server_local_by_cat"].get(cat, 0.0))
            server_combined_by_cat_all[cat].append(extra_stats["server_combined_by_cat"].get(cat, 0.0))
            server_hit_by_cat_all[cat].append(extra_stats["server_hit_by_cat"].get(cat, 0.0))
            server_queue_by_cat_all[cat].append(extra_stats["server_queue_by_cat"].get(cat, 0.0))

        for j in range(NUM_UAVS):
            server_hit_by_uav_all[j].append(extra_stats["server_hit_by_uav"].get(j, 0.0))
            server_sim_by_uav_all[j].append(extra_stats["server_sim_by_uav"].get(j, 0.0))


        user_avg_local_list.append(ep_stats["user_avg_local"])
        user_avg_global_list.append(ep_stats["user_avg_global"])
        user_avg_combined_list.append(ep_stats["user_avg_combined"])
        server_avg_local_list.append(ep_stats["server_avg_local"])
        server_avg_global_list.append(ep_stats["server_avg_global"])
        server_avg_combined_list.append(ep_stats["server_avg_combined"])
        server_discount_list.append(ep_stats["overall_discount"])
        uav_queue_history_all.append(np.mean(env.server_queue_history[-1]))

        server_hit_rate_ep = np.mean([uav['stats_history'][-1].get('hit_rate', 0.0) if uav['stats_history'] else 0.0 for uav in env.uavs])
        server_hit_rate_list.append(server_hit_rate_ep)

        user_loss, server_loss, server_hess = update_agents(env, server_buffers, server_agent)
        user_critic_loss_list.append(0)
        server_critic_loss_list.append(server_loss[1])

        # Record the current number of tasks at the end of the episode
        task_count_over_episodes.append(0)


    # Assemble final statistics
    final_stats = {
        "user_critic_loss": user_critic_loss_list,
        "server_critic_loss": server_critic_loss_list,
        "user_local_reward": user_avg_local_list,
        "user_global_reward": user_avg_global_list,
        "user_combined_reward": user_avg_combined_list,
        "server_local_reward": server_avg_local_list,
        "server_global_reward": server_avg_global_list,
        "server_combined_reward": server_avg_combined_list,
        "overall_discount": server_discount_list,
        "server_hit_rate": [np.mean(hr) for hr in server_hit_rate_list],
        "avg_server_queue": uav_queue_history_all[-1]
    }

    # Assemble the extra per-category statistics
    extra_stats_final = {
        "user_local_by_env": user_local_by_env_all,
        "user_combined_by_env": user_combined_by_env_all,
        "server_local_by_cat": server_local_by_cat_all,
        "server_combined_by_cat": server_combined_by_cat_all,
        "server_hit_by_uav": server_hit_by_uav_all,
        "server_sim_by_uav": server_sim_by_uav_all,
        "server_hit_by_cat": server_hit_by_cat_all,
        "server_queue_by_cat": server_queue_by_cat_all,
        "server_feature_history": server_feature_history,
        "uav_categories": uav_categories,
        "task_count_over_episodes": task_count_over_episodes
    }

    final_metrics = {
        "user_global_reward": final_stats["user_global_reward"][-1],
        "user_local_reward": final_stats["user_local_reward"][-1],
        "user_combined_reward": final_stats["user_combined_reward"][-1],
        "server_global_reward": final_stats["server_global_reward"][-1],
        "server_local_reward": final_stats["server_local_reward"][-1],
        "server_combined_reward": final_stats["server_combined_reward"][-1],
        "overall_discount": final_stats["overall_discount"][-1],
        "server_hit_rate": final_stats["server_hit_rate"],
        "avg_server_queue": final_stats["avg_server_queue"]
    }

    print("[RUN] All episodes done.")


    return final_stats, final_metrics, extra_stats_final


def run_transfer_learning_baseline(enable_server_caching=True, enable_similarity=True, case_label="transfer_baseline"):
    print(f"[RUN] Transfer Learning Baseline (server_caching={enable_server_caching}, similarity={enable_similarity})")
    print(f"[RUN] Using FULL TEST environment with complete complexity (Fine-tuning Mode)")

    # 1. Base test-environment parameters
    # TEST_PARAMS and TEST_PARAMS_LIST are module-level configurations
    test_steps = TEST_PARAMS["steps_per_episode"]
    num_episodes = 100

    # 2. Initialize the agent
    server_agent = ServerMAPPOAgent(
        num_servers=NUM_UAVS, obs_dim=200, action_dim=100, hidden_dim=32,
        lr=1e-4, gamma=0.99, lam=0.95, alpha=0.5, beta=0.5, device=device,
        clip_ratio=0.4, entropy_coef=0.1, max_grad_norm=0.5
    ) if enable_server_caching else None

    # Parameter paths (loaded from the pretraining results)
    load_dir = "./pretrained_params"

    # 3. [Key step] Load pretrained parameters (starting point of transfer learning)
    # Transfer learning always starts from the pretrained parameters
    if server_agent:
        print(f"[INIT] Loading pretrained parameters from {load_dir}...")
        for j in range(NUM_UAVS):
            # Load actor
            actor_path = os.path.join(load_dir, f"server_actor_{j}_non_policy_params.pt")
            if os.path.exists(actor_path):
                server_agent.actors[j].load_state_dict(torch.load(actor_path), strict=False)
                print(f"  -> Loaded Pretrained Actor {j}")
            else:
                print(f"  [WARNING] Pretrained Actor {j} not found at {actor_path}")

            # Load critic
            critic_path = os.path.join(load_dir, f"server_critic_{j}_params.pt")
            if os.path.exists(critic_path):
                server_agent.critics[j].load_state_dict(torch.load(critic_path), strict=False)
                print(f"  -> Loaded Pretrained Critic {j}")
            else:
                print(f"  [WARNING] Pretrained Critic {j} not found at {critic_path}")

        # Reset hidden states before starting the new task
        server_agent.reset_hidden_states()

    # ==================================================================================
    # [Diff] All DC-ELLA learner initialization and related logic is removed
    # ==================================================================================

    # 4. Statistics containers (same structure as run_mappo_test)
    user_avg_local_list, user_avg_global_list, user_avg_combined_list = [], [], []
    server_avg_local_list, server_avg_global_list, server_avg_combined_list = [], [], []
    server_discount_list, user_critic_loss_list, server_critic_loss_list, server_hit_rate_list = [], [], [], []
    task_count_over_episodes = []  # kept for format compatibility, filled with 0

    # Helper bookkeeping variables
    task_occurrence_log = defaultdict(list)
    policy_change_events = []

    # User-type list
    user_types = TEST_PARAMS["USER_TYPES"]
    server_categories = [1, 2, 3, 4, 5]

    # Detailed statistics containers
    user_local_by_env_all = {env: [] for env in user_types}
    user_combined_by_env_all = {env: [] for env in user_types}
    server_local_by_cat_all = {cat: [] for cat in server_categories}
    server_combined_by_cat_all = {cat: [] for cat in server_categories}
    server_hit_by_uav_all = {j: [] for j in range(NUM_UAVS)}
    server_sim_by_uav_all = {j: [] for j in range(NUM_UAVS)}
    server_hit_by_cat_all = {cat: [] for cat in server_categories}
    server_queue_by_cat_all = {cat: [] for cat in server_categories}
    server_feature_history = {j: [] for j in range(NUM_UAVS)}

    uav_queue_history_all, uav_categories = [], None
    previous_task_ids = {j: None for j in range(NUM_UAVS)}  # kept for compatibility only, unused

    # ==================================================================================
    # 5. Main loop (transfer learning)
    # ==================================================================================
    for ep in range(num_episodes):

        print(f"[RUN] Episode {ep}/{num_episodes}")

        # [Environment switching] identical to run_mappo_test
        # Version label for each environment update
        version = ep // ENV_SWITCH_EPISODES + 1

        # Switch the environment parameters every ENV_SWITCH_EPISODES episodes
        param_idx = (ep // ENV_SWITCH_EPISODES) % len(TEST_PARAMS_LIST)
        current_test_params = TEST_PARAMS_LIST[param_idx]

        # Re-create the environment
        env = MultiUserEnv(
            episode_length=current_test_params["steps_per_episode"], time_slot=1.0, alpha=1, beta=0,
            NUM_USERS=NUM_USERS, NUM_UAVS=NUM_UAVS, task_rates=task_rates, task_prob=task_prob, TEST_PARAMS=TEST_PARAMS,
            MAPSIZE=MAPSIZE, resolution_requirement_user=resolution_requirement_user,
            time_requirement_user=time_requirement_user, TASK_SIZE=TASK_SIZE, UAV_HEIGHT=UAV_HEIGHT,
            subject_similarity_matrix=subject_similarity_matrix, object_similarity_matrix=object_similarity_matrix,
            BIT_PER_CYCLE=BIT_PER_CYCLE,
            mu_compute=mu_compute, mu_downlink=mu_downlink, COMPRESSION_RATE=COMPRESSION_RATE,
            channel_params=channel_params, bandwidth_map_user=bandwidth_map_user, CARRIER_FREQ=CARRIER_FREQ,
            P_USER=P_USER, G_USER=G_USER, NOISE_POWER=NOISE_POWER, P_UAV=P_UAV, G_UAV=G_UAV,
            image_label_mapping=image_label_mapping,
            enable_server_caching=enable_server_caching,
            enable_similarity=enable_similarity,
            env_params=current_test_params
        )

        print(f"[ENV RE-CREATED] Episode {ep}, new params applied (Index {param_idx}).")

        obs_list = env.reset()

        # Sync helper variables
        user_types = current_test_params["USER_TYPES"]
        test_steps = current_test_params["steps_per_episode"]

        if uav_categories is None:
            uav_categories = [uav['category'] for uav in env.uavs]

        if server_agent:
            server_agent.reset_hidden_states()

        # ==================================================================================
        # [Diff] GUIDANCE-1 (DC-ELLA initial policy guidance) is removed
        # Transfer learning simply continues from the current network parameters, no reset or guidance
        # ==================================================================================

        # 6. Data collection (interaction)
        user_buffers, server_buffers, ep_stats, extra_stats, server_feature_avg = collect_episode_data(
            env, server_agent, test_steps, version
        )

        # 7. Logging
        # (1) Feature history
        for j in range(NUM_UAVS):
            server_feature_history[j].append(server_feature_avg[j])

        # (2) Accumulate user statistics (user_types may change dynamically)
        for env_type in user_types:
            val_local = extra_stats["user_local_by_env"].get(env_type, 0.0)
            val_combined = extra_stats["user_combined_by_env"].get(env_type, 0.0)

            if env_type not in user_local_by_env_all:
                user_local_by_env_all[env_type] = []
                user_combined_by_env_all[env_type] = []

            user_local_by_env_all[env_type].append(val_local)
            user_combined_by_env_all[env_type].append(val_combined)

        # (3) Accumulate server-category statistics
        for cat in server_categories:
            server_local_by_cat_all[cat].append(extra_stats["server_local_by_cat"].get(cat, 0.0))
            server_combined_by_cat_all[cat].append(extra_stats["server_combined_by_cat"].get(cat, 0.0))
            server_hit_by_cat_all[cat].append(extra_stats["server_hit_by_cat"].get(cat, 0.0))
            server_queue_by_cat_all[cat].append(extra_stats["server_queue_by_cat"].get(cat, 0.0))

        # (4) Accumulate per-UAV server statistics
        for j in range(NUM_UAVS):
            server_hit_by_uav_all[j].append(extra_stats["server_hit_by_uav"].get(j, 0.0))
            server_sim_by_uav_all[j].append(extra_stats["server_sim_by_uav"].get(j, 0.0))

        # (5) Accumulate scalar lists
        user_avg_local_list.append(ep_stats["user_avg_local"])
        user_avg_global_list.append(ep_stats["user_avg_global"])
        user_avg_combined_list.append(ep_stats["user_avg_combined"])
        server_avg_local_list.append(ep_stats["server_avg_local"])
        server_avg_global_list.append(ep_stats["server_avg_global"])
        server_avg_combined_list.append(ep_stats["server_avg_combined"])
        server_discount_list.append(ep_stats["overall_discount"])
        uav_queue_history_all.append(np.mean(env.server_queue_history[-1]))

        server_hit_rate_ep = np.mean(
            [uav['stats_history'][-1].get('hit_rate', 0.0) if uav['stats_history'] else 0.0 for uav in env.uavs])
        server_hit_rate_list.append(server_hit_rate_ep)

        # 8. [Core update] PPO update only (fine-tuning)
        # Gradients come from the current episode's data and fine-tune the network parameters
        if server_agent:
            user_loss, server_loss, server_hess = update_agents(env, server_buffers, server_agent)
            server_critic_loss_list.append(server_loss[1])
        else:
            server_critic_loss_list.append(0)

        user_critic_loss_list.append(0)

        # ==================================================================================
        # [Diff] GUIDANCE-2 (periodic DC-ELLA update of L and D) is removed
        # In transfer-learning mode, the policy only changes through update_agents
        # ==================================================================================

        print(f"[INFO] Transfer Episode {ep} completed.")

        # Keep the task_count list length consistent, fill with 0
        task_count_over_episodes.append(0)

    # 9. Build the final return value
    final_stats = {
        "user_critic_loss": user_critic_loss_list,
        "server_critic_loss": server_critic_loss_list,
        "user_local_reward": user_avg_local_list,
        "user_global_reward": user_avg_global_list,
        "user_combined_reward": user_avg_combined_list,
        "server_local_reward": server_avg_local_list,
        "server_global_reward": server_avg_global_list,
        "server_combined_reward": server_avg_combined_list,
        "overall_discount": server_discount_list,
        "server_hit_rate": [np.mean(hr) for hr in server_hit_rate_list],
        "avg_server_queue": uav_queue_history_all[-1] if len(uav_queue_history_all) > 0 else 0
    }

    extra_stats_final = {
        "user_local_by_env": user_local_by_env_all,
        "user_combined_by_env": user_combined_by_env_all,
        "server_local_by_cat": server_local_by_cat_all,
        "server_combined_by_cat": server_combined_by_cat_all,
        "server_hit_by_uav": server_hit_by_uav_all,
        "server_sim_by_uav": server_sim_by_uav_all,
        "server_hit_by_cat": server_hit_by_cat_all,
        "server_queue_by_cat": server_queue_by_cat_all,
        "server_feature_history": server_feature_history,
        "uav_categories": uav_categories,
        # Compatibility fields
        "task_count_over_episodes": task_count_over_episodes,
        "task_occurrence_log": dict(task_occurrence_log),
        "policy_change_events": policy_change_events
    }

    final_metrics = {
        "user_global_reward": final_stats["user_global_reward"][-1] if final_stats["user_global_reward"] else 0,
        "user_local_reward": final_stats["user_local_reward"][-1] if final_stats["user_local_reward"] else 0,
        "user_combined_reward": final_stats["user_combined_reward"][-1] if final_stats["user_combined_reward"] else 0,
        "server_global_reward": final_stats["server_global_reward"][-1] if final_stats["server_global_reward"] else 0,
        "server_local_reward": final_stats["server_local_reward"][-1] if final_stats["server_local_reward"] else 0,
        "server_combined_reward": final_stats["server_combined_reward"][-1] if final_stats[
            "server_combined_reward"] else 0,
        "overall_discount": final_stats["overall_discount"][-1] if final_stats["overall_discount"] else 0,
        "server_hit_rate": final_stats["server_hit_rate"],
        "avg_server_queue": final_stats["avg_server_queue"]
    }

    print("[RUN] Transfer Learning Baseline completed all episodes.")

    # [Diff] Saving to task_analyse is removed here; run_case_transfer_learning handles it

    return final_stats, final_metrics, extra_stats_final

def run_case(case_params):
    set_seed(case_params["seed"])

    # Run pretraining first
    final_stats_pretrain, final_metrics_pretrain, extra_stats_pretrain = run_mappo_pretrain(
        enable_server_caching=case_params["enable_server_caching"],
        enable_similarity=case_params["enable_similarity"],
        case_label=f"{case_params['label']}_pretrain"
    )

    # Save pretraining data
    save_dir = "plot_data_cases"
    os.makedirs(save_dir, exist_ok=True)
    case_data_pretrain = {
        "final_stats": final_stats_pretrain,
        "final_metrics": final_metrics_pretrain,
        "extra_stats": extra_stats_pretrain
    }
    save_path_pretrain = os.path.join(save_dir, f"{case_params['label']}_pretrain_data.pkl")
    with open(save_path_pretrain, "wb") as f:
        pickle.dump(case_data_pretrain, f)
    print(f"[PRETRAIN DONE] data saved to {save_path_pretrain}")

    # Run the test phase
    final_stats_test, final_metrics_test, extra_stats_test = run_mappo_test(
        enable_server_caching=case_params["enable_server_caching"],
        enable_similarity=case_params["enable_similarity"],
        case_label=f"{case_params['label']}_test",
        get_pretrain_params=True  # explicitly use the pretrained parameters
    )

    # Save test-phase data
    case_data_test = {
        "final_stats": final_stats_test,
        "final_metrics": final_metrics_test,
        "extra_stats": extra_stats_test
    }
    save_path_test = os.path.join(save_dir, f"{case_params['label']}_test_data.pkl")
    with open(save_path_test, "wb") as f:
        pickle.dump(case_data_test, f)
    print(f"[TEST DONE] data saved to {save_path_test}")

    return {
        "label": case_params["label"],
        "pretrain": {
            "final_stats": final_stats_pretrain,
            "final_metrics": final_metrics_pretrain,
            "extra_stats": extra_stats_pretrain
        },
        "test": {
            "final_stats": final_stats_test,
            "final_metrics": final_metrics_test,
            "extra_stats": extra_stats_test
        }
    }


def run_case_with_pretrain_and_test(case_params):
    set_seed(case_params["seed"])

    save_dir = "plot_data_cases"
    os.makedirs(save_dir, exist_ok=True)
    pretrain_save_path = os.path.join(save_dir, f"{case_params['label']}_pretrain_data.pkl")

    # Check whether pretraining data already exists
    if os.path.exists(pretrain_save_path):
        with open(pretrain_save_path, "rb") as f:
            pretrain_data = pickle.load(f)
        print(f"[CASE PRETRAIN LOADED] loaded pretraining data {pretrain_save_path}")
    else:
        # Run pretraining + save parameters
        pretrain_stats, pretrain_metrics, pretrain_extra_stats = run_mappo_pretrain(
            enable_server_caching=case_params["enable_server_caching"],
            enable_similarity=case_params["enable_similarity"],
            case_label=f"{case_params['label']}_pretrain"
        )

        pretrain_data = {
            "final_stats": pretrain_stats,
            "final_metrics": pretrain_metrics,
            "extra_stats": pretrain_extra_stats
        }
        with open(pretrain_save_path, "wb") as f:
            pickle.dump(pretrain_data, f)
        print(f"[CASE PRETRAIN DONE] data saved to {pretrain_save_path}")

    # Run the test phase
    test_stats, test_metrics, test_extra_stats = run_mappo_test(
        enable_server_caching=case_params["enable_server_caching"],
        enable_similarity=case_params["enable_similarity"],
        case_label=f"{case_params['label']}_test",
        get_pretrain_params=True
    )

    # Save test data
    test_data = {
        "final_stats": test_stats,
        "final_metrics": test_metrics,
        "extra_stats": test_extra_stats
    }
    test_save_path = os.path.join(save_dir, f"{case_params['label']}_test_data.pkl")
    with open(test_save_path, "wb") as f:
        pickle.dump(test_data, f)
    print(f"[CASE TEST DONE] data saved to {test_save_path}")

    return {
        "label": case_params["label"],
        "pretrain": pretrain_data,
        "test": test_data
    }

def run_case_base_test(case_params):
    set_seed(case_params["seed"])

    save_dir = "plot_data_cases"
    os.makedirs(save_dir, exist_ok=True)


    # Run the test phase
    base_stats, base_metrics, base_extra_stats = run_mappo_base_test(
        enable_server_caching=case_params["enable_server_caching"],
        enable_similarity=case_params["enable_similarity"],
        case_label=f"{case_params['label']}_test",
        get_pretrain_params=True
    )

    # Save test data
    base_test_data = {
        "final_stats": base_stats,
        "final_metrics": base_metrics,
        "extra_stats": base_extra_stats
    }
    test_save_path = os.path.join(save_dir, f"{case_params['label']}_base_test_data.pkl")
    with open(test_save_path, "wb") as f:
        pickle.dump(base_test_data, f)
    print(f"[CASE TEST DONE] data saved to {test_save_path}")

    return {
        "label": case_params["label"],
        "base_test": base_test_data
    }



def run_case_transfer_learning(case_params):
    set_seed(case_params["seed"])

    save_dir = "plot_data_cases"
    os.makedirs(save_dir, exist_ok=True)

    # 1. Check pretraining data (transfer learning also depends on the pretrained parameters)
    # If pretraining has already run, load its data; otherwise run pretraining first
    # Note: run_transfer_learning_baseline actually depends on the .pt parameter files on disk
    # loading the .pkl here mainly ensures that pretraining has been executed
    pretrain_save_path = os.path.join(save_dir, f"{case_params['label']}_pretrain_data.pkl")

    if os.path.exists(pretrain_save_path):
        with open(pretrain_save_path, "rb") as f:
            pretrain_data = pickle.load(f)
        print(f"[CASE TL PRETRAIN LOADED] loaded pretraining data {pretrain_save_path}")
    else:
        # Without pretraining data, pretraining must run first to produce the parameter files
        print(f"[CASE TL PRETRAIN MISSING] pretraining data not found, running pretraining...")
        pretrain_stats, pretrain_metrics, pretrain_extra_stats = run_mappo_pretrain(
            enable_server_caching=case_params["enable_server_caching"],
            enable_similarity=case_params["enable_similarity"],
            case_label=f"{case_params['label']}_pretrain"
        )

        pretrain_data = {
            "final_stats": pretrain_stats,
            "final_metrics": pretrain_metrics,
            "extra_stats": pretrain_extra_stats
        }
        with open(pretrain_save_path, "wb") as f:
            pickle.dump(pretrain_data, f)
        print(f"[CASE TL PRETRAIN DONE] data saved to {pretrain_save_path}")

    # 2. Run the test phase of the transfer-learning baseline
    tl_stats, tl_metrics, tl_extra_stats = run_transfer_learning_baseline(
        enable_server_caching=case_params["enable_server_caching"],
        enable_similarity=case_params["enable_similarity"],
        case_label=f"{case_params['label']}_transfer_baseline"
    )

    # 3. Save the transfer-learning test data
    # Note: the file name carries a _transfer_baseline suffix to distinguish it
    tl_data = {
        "final_stats": tl_stats,
        "final_metrics": tl_metrics,
        "extra_stats": tl_extra_stats
    }

    # e.g. plot_data_cases/default_transfer_baseline_data.pkl
    tl_save_path = os.path.join(save_dir, f"{case_params['label']}_transfer_baseline_data.pkl")

    with open(tl_save_path, "wb") as f:
        pickle.dump(tl_data, f)
    print(f"[CASE TL TEST DONE] data saved to {tl_save_path}")

    return {
        "label": case_params["label"],
        "pretrain": pretrain_data,
        "transfer_baseline": tl_data
    }


def parallel_compare_conditions():
    cases = [
        # {"label": "Only_Hit", "enable_server_caching": True, "enable_similarity": False, "seed": 3},
        {"label": "Hit+Similarity", "enable_server_caching": True, "enable_similarity": True, "seed": 3},
        # {"label": "No_Cache", "enable_server_caching": False, "enable_similarity": False, "seed": 3},
    ]

    # Run all cases in parallel (uses the top-level functions)
    with multiprocessing.Pool(processes=max(len(cases), 1)) as pool:
        # results_1 = pool.map(run_case_with_pretrain_and_test, cases)
        # results_2 = pool.map(run_case_base_test, cases)
        results_3 = pool.map(run_case_transfer_learning, cases)

    print("[ALL CASES COMPLETED]")
    return  results_3

if __name__ == '__main__':
    parallel_compare_conditions()


# # Comparison function: run the different modes sequentially under the same seed
# def compare_modes():
#     seed = 1
#     # print("=== Running hit only (enable_similarity=False) ===")
#     # set_seed(seed)
#     # run_mappo(enable_server_caching=True, enable_similarity=False)
#     # print("=== Running hit + similarity (enable_similarity=True) ===")
#     # set_seed(seed)
#     # run_mappo(enable_server_caching=True, enable_similarity=True)
#     print("=== Running without server caching (server not trained) ===")
#     set_seed(seed)
#     run_mappo(enable_server_caching=False, enable_similarity=False)
#
# if __name__=="__main__":
#     # Call the comparison function to run the modes sequentially under the same seed
#     compare_modes()
