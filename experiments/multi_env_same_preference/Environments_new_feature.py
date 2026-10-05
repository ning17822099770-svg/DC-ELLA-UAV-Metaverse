"""Simulation environment and MDP of the UAV-assisted semantic Metaverse.

Implements the system model (Sec. II) and the semantic-aware caching and
rendering schemes (Sec. III):
  * users generate semantic requests (subject/object labels of Visual Genome
    images, see image_label_mapping.pkl) with Poisson arrivals per environment;
  * A2G uplink/downlink rates (Eq. 2) and rendering time/energy (Eq. 3);
  * cache hit, cache reuse based on the reuse probability P^re (Eq. 11-12),
    and full computation, giving the computational saving S^ave (Eq. 13-14);
  * task completion quality Sa_j(t) (Eq. 15) and server utility U_j(t),
    which is the reward (Eq. 16);
  * server observations (state, Sec. V-A) and the task feature vector Phi used
    by DC-ELLA for lifelong task detection (compute_features_for_tasks).

Main class: MultiUserEnv. `enable_server_caching` / `enable_similarity` select
No-Cache, Only-Hit (hit only) or Hit+Similarity (hit + semantic reuse).

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

def preprocess_feature_vector(freq_part, subj_avg, obj_avg,
                              resolution, cpu_freq,
                              hist_res, hist_cpu):
    # 1) Log1p of request frequencies
    freq = np.log1p(np.array(freq_part, dtype=np.float32))

    # 2) Min-max scaling of similarities
    sim = np.concatenate([subj_avg, obj_avg], axis=0)
    s_min, s_max = sim.min(), sim.max()
    if s_max > s_min:
        sim = (sim - s_min) / (s_max - s_min)

    # 3) Z-score of resolution & CPU frequency
    res_mu, res_std = np.mean(hist_res), np.std(hist_res) + 1e-8
    cpu_mu, cpu_std = np.mean(hist_cpu), np.std(hist_cpu) + 1e-8
    res_n = (resolution - res_mu) / res_std
    cpu_n = (cpu_freq   - cpu_mu)   / cpu_std

    # 4) Concatenate & L2-normalize
    vec = np.concatenate([freq, sim, [res_n, cpu_n]], axis=0).astype(np.float32)
    norm = np.linalg.norm(vec) + 1e-8
    return vec / norm


    # def preprocess_feature_vector(
    #         freq_part: np.ndarray,
    #         subj_avg: np.ndarray,
    #         obj_avg: np.ndarray,
    #         resolution: float,
    #         cpu_freq: float,
    #         hist_res: np.ndarray,
    #         hist_cpu: np.ndarray,
    #         beta_sim: float = 5.0,
    #         eps: float = 1e-8
    # ) -> np.ndarray:
    #     """
    #     Block-wise preprocessing of a single UAV's feature vector:
    #       - freq_part:      list/array of length 100, raw request-frequency counts
    #       - subj_avg,obj_avg: length 50 each, row-averaged similarities
    #       - resolution:     scalar (pixel ratio already normalized to [0,1])
    #       - cpu_freq:       scalar (in GHz)
    #       - hist_res,hist_cpu: resolution & CPU frequency over all past episodes, used for z-score
    #
    #     Returns the concatenated vector, length = 100 + 100 + 2 = 202.
    #     """
    #     # 1) Only apply log1p to frequencies to keep magnitude differences
    #     freq = np.log1p(np.asarray(freq_part, dtype=np.float32))
    #     # (Optional) to up-weight the frequencies:
    #     # freq *= beta_freq
    #
    #     # 2) Z-score the similarities, then weight them
    #     sim = np.concatenate([subj_avg, obj_avg], axis=0).astype(np.float32)
    #     mu_sim, std_sim = sim.mean(), sim.std() + eps
    #     sim = (sim - mu_sim) / std_sim
    #     sim = beta_sim * sim
    #
    #     # 3) Z-score resolution & CPU
    #     hist_res = np.asarray(hist_res, dtype=np.float32)
    #     mu_res, std_res = hist_res.mean(), hist_res.std() + eps
    #     res_n = (resolution - mu_res) / std_res
    #
    #     hist_cpu = np.asarray(hist_cpu, dtype=np.float32)
    #     mu_cpu, std_cpu = hist_cpu.mean(), hist_cpu.std() + eps
    #     cpu_n = (cpu_freq - mu_cpu) / std_cpu
    #
    #     # 4) Concatenate and return
    #     vec = np.concatenate([freq, sim, np.array([res_n, cpu_n], dtype=np.float32)], axis=0)
    #     return vec

class MultiUserEnv:
    def __init__(self, episode_length=30, time_slot=1.0, alpha=0.8, beta=0.2,
                 NUM_USERS=None, NUM_UAVS=None, task_rates=None, task_prob=None, TEST_PARAMS=None,
                 MAPSIZE=None, resolution_requirement_user=None, time_requirement_user=None, TASK_SIZE=None, UAV_HEIGHT=None,
                 subject_similarity_matrix=None, object_similarity_matrix=None, BIT_PER_CYCLE=None, mu_compute=None, mu_downlink=None,
                 COMPRESSION_RATE = None, channel_params=None, bandwidth_map_user=None, CARRIER_FREQ=None,
                 P_USER=None, G_USER=None, NOISE_POWER=None, P_UAV=None, G_UAV=None, image_label_mapping=None,
                 enable_server_caching=True,
                 enable_similarity=True,
                 env_params=None):  # environment-parameter dict
        # Use the given environment parameters, or the defaults

        if env_params is None:
            env_params = TEST_PARAMS  # default to the test-environment parameters
        # Read values from the environment parameters
        self.num_users = env_params.get("NUM_USERS", NUM_USERS)
        self.user_types = env_params.get("USER_TYPES", ['Entertainment', 'Home', 'Wild', 'Traffic', 'Work'])
        self.task_rates = env_params.get("task_rates", task_rates)
        self.task_prob = env_params.get("task_prob", task_prob)
        # Other original parameters
        self.episode_length = episode_length
        self.time_slot = time_slot
        self.alpha = alpha
        self.beta = beta
        self.num_agents = self.num_users
        self.action_dim = NUM_UAVS
        self.obs_dim_per_agent = 1 + NUM_UAVS * (4)
        self.current_step_in_episode = 0
        self.global_time = 0
        self.enable_server_caching = enable_server_caching
        self.enable_similarity = enable_similarity  # [MODIFIED]
        self.MAPSIZE = MAPSIZE
        self.NUM_UAVS = NUM_UAVS
        self.BIT_PER_CYCLE = BIT_PER_CYCLE
        self.mu_compute = mu_compute
        self.mu_downlink = mu_downlink
        self.COMPRESSION_RATE = COMPRESSION_RATE
        self.channel_params = channel_params
        self.bandwidth_map_user = bandwidth_map_user
        self.CARRIER_FREQ = CARRIER_FREQ
        self.P_USER = P_USER
        self.G_USER = G_USER
        self.NOISE_POWER = NOISE_POWER
        self.P_UAV = P_UAV
        self.G_UAV = G_UAV
        self.image_label_mapping = image_label_mapping
        self.resolution_requirement_user = resolution_requirement_user
        self.time_requirement_user = time_requirement_user
        self.UAV_HEIGHT = UAV_HEIGHT
        self.TASK_SIZE = TASK_SIZE
        self.subject_similarity_matrix = subject_similarity_matrix
        self.object_similarity_matrix = object_similarity_matrix

        # self.users = initialize_users()
        # self.uavs = initialize_uavs(enable_server_caching=enable_server_caching)
        # self.v_uav_compute = [0] * NUM_UAVS
        # self.v_uav_downlink_user = [[0] * NUM_USERS for _ in range(NUM_UAVS)]
        # self.v_user_requests = [0] * NUM_USERS
        # self.v_queue_tasks = [{i: [] for i in range(self.num_agents)} for _ in range(NUM_UAVS)]
        # Initialize users according to the configured user types and count
        self.users = self._initialize_users()
        self.uavs = self.initialize_uavs(enable_server_caching=enable_server_caching)
        self.v_uav_compute = [0] * self.NUM_UAVS
        self.v_uav_downlink_user = [[0] * self.num_users for _ in range(self.NUM_UAVS)]
        self.v_user_requests = [0] * self.num_users
        self.v_queue_tasks = [{i: [] for i in range(self.num_agents)} for _ in range(self.NUM_UAVS)]

        # --- ➊ similarity-matrix cache  -----------------------------------
        # Similarity cache dictionary
        self._uav_similarity_cache = {}  # key: uav_id → {'sig': tuple, 'val': dict}

        # Storage for the features of the most recently uploaded tasks (optional, for analysis)
        self.latest_transmitted_features = {j: [] for j in range(self.NUM_UAVS)}

        # Initialize online historical statistics here if needed
        self.historical_resolutions = []
        self.historical_cpu_freqs   = []


    def _initialize_users(self):    # 2025 0401 1420
        """Initialize users according to the current environment configuration."""
        positions = np.random.rand(self.num_users, 2) * self.MAPSIZE
        env_types = self.user_types
        base = self.num_users // len(env_types)
        remainder = self.num_users % len(env_types)
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
                    'resolution_req' : self.resolution_requirement_user[e_idx],
                    'time_req' : self.time_requirement_user[e_idx],
                    'requests': [],
                    'received': []
                }
                users.append(user)
                user_index += 1
        print(f"[INIT] Initialize {len(users)} users with environments: {env_types}")
        return users

    def initialize_uavs(self, enable_server_caching=True):
        bandwidth_map_server = [50e6, 30e6, 25e6, 15e6, 10e6]
        resolutions_UAV = [1280 * 720, 1920 * 1080, 2048 * 1080, 3072 * 1620, 4096 * 2048]
        cpu_freqs = [1e9, 1.5e9, 2e9, 2.5e9, 3e9]
        rows = int(math.ceil(math.sqrt(self.NUM_UAVS)))
        cols = int(math.ceil(self.NUM_UAVS / rows))
        delta_x = self.MAPSIZE / (cols + 1)
        delta_y = self.MAPSIZE / (rows + 1)
        positions = []
        count = 0
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                if count < self.NUM_UAVS:
                    x = c * delta_x
                    y = r * delta_y
                    positions.append((x, y))
                    count += 1
                else:
                    break
        uavs = []
        for i in range(self.NUM_UAVS):
            cat_idx = i % 5
            # Subject cache uses indices 1-50 and object cache uses 51-100; randomly pick 10 contents for each
            prev_subj_cache = random.sample(range(1, 51), 10)
            prev_obj_cache = random.sample(range(51, 101), 10)
            if enable_server_caching:
                cur_subj_cache = []
                cur_obj_cache = []
            else:
                # When server caching is disabled, the caches are empty
                cur_subj_cache = []
                cur_obj_cache = []
            uav = {
                'id': i + 1,
                'category': cat_idx + 1,
                'position': (*positions[i], self.UAV_HEIGHT),
                'cpu_freq': cpu_freqs[cat_idx],
                'resolution': resolutions_UAV[cat_idx],
                'bandwidth': bandwidth_map_server[cat_idx],
                'queue': [],
                'downlink_queue': [],
                'subject_cache': cur_subj_cache,
                'object_cache': cur_obj_cache,
                'prev_subject_cache': prev_subj_cache,
                'prev_object_cache': prev_obj_cache,
                'cache_hits': 0,
                'total_tasks_processed': 0,
                'cache_discount_acc': 0.0,
                'stats_history': []
            }
            uavs.append(uav)
        print(f"[INIT] Initialize {self.NUM_UAVS} UAV servers.")
        return uavs

    def reset(self):
        print("\n[ENV] Reset environment ...")
        self.current_step_in_episode = 0
        self.global_time = 0
        # self.users = initialize_users()
        self.users = self._initialize_users()  # use the environment-aware initializer
        self.uavs = self.initialize_uavs(enable_server_caching=self.enable_server_caching)
        self.v_uav_compute = [0] * self.NUM_UAVS
        self.v_uav_downlink_user = [[0] * self.num_users for _ in range(self.NUM_UAVS)]
        self.v_user_requests = [0] * self.num_users

        self._uav_similarity_cache.clear()  # queues were refreshed, invalidate the cache

        return self.get_global_obs()

    def _generate_tasks_for_user(self, user, image_label_mapping): # 0401 1420
        e = user['environment']
        # if np.random.rand() > self.task_prob.get(e, 0.5):  # probability from the current environment config
        #     return
        lam = self.task_rates.get(e)  # Poisson arrival rate from the current environment config
        num_tasks = np.random.poisson(lam)
        if num_tasks <= 0:
            return
        idx_env = user['environment_index']
        resolution_req = self.resolution_requirement_user[idx_env]
        time_req = self.time_requirement_user[idx_env]
        for _ in range(num_tasks):
            if idx_env == 0:
                image_index = random.randint(1, 10)
            elif idx_env == 1:
                image_index = random.randint(11, 20)
            elif idx_env == 2:
                image_index = random.randint(21, 30)
            elif idx_env == 3:
                image_index = random.randint(31, 40)
            else:
                image_index = random.randint(41, 50)
            triple_info = image_label_mapping.get(image_index, {"subject_indices": [], "object_indices": []})
            subject_indices = triple_info.get("subject_indices", [])[:3]
            object_indices = triple_info.get("object_indices", [])[:3]
            min_len = min(len(subject_indices), len(object_indices))
            if min_len == 0:
                continue
            new_task = {
                'task_id': f"user{user['id']}_t{self.global_time}",
                'user_id': user['id'],
                'time_created': self.global_time,
                'image_index': image_index,
                'subject_list': subject_indices[:min_len],
                'object_list': [obj + 50 for obj in object_indices[:min_len]],
                'num_triples': min_len,
                'size': self.TASK_SIZE,
                'resolution_req': resolution_req,
                'time_req': time_req
            }
            user['requests'].append(new_task)
        print('length of user =', len(user['requests']))

    def get_global_obs(self):
        obs_list = [self._get_local_obs(i) for i in range(self.num_agents)]
        return obs_list

    def _get_local_obs(self, user_i):
        obs = []
        obs.append(float(len(self.users[user_i]['requests'])))
        for j in range(self.NUM_UAVS):
            obs.append(float(self.v_uav_compute[j]))
            obs.append(float(self.v_uav_downlink_user[j][user_i]))
            obs.append(float(self.uavs[j]['resolution'] / (1280.0 * 720.0)))
            obs.append(float(self.uavs[j]['cpu_freq'] / (1e8)))
        return np.array(obs, dtype=np.float32)

    def compute_uav_content_metrics(self):
        """
        Count the content indices of the tasks in each UAV queue:
          subject: use the index directly (1-50)
          object: map the raw object index (51-100) to 1-50
        """
        metrics = {}
        for uav in self.uavs:
            subj_stats = {}
            obj_stats = {}
            total_subj = 0
            total_obj = 0
            for task in uav['queue']:
                user_id = task.get('user_id', None)
                if user_id is None:
                    continue
                user = self.users[user_id - 1]
                arrival_rate = self.task_prob[user['environment']]
                time_req = task.get('time_req', 0.0)
                res_req = task.get('resolution_req', 0.0)
                for s in task.get('subject_list', []):
                    total_subj += 1
                    if s not in subj_stats:
                        subj_stats[s] = [0, 0.0, 0.0, 0.0]
                    subj_stats[s][0] += 1
                    subj_stats[s][1] += arrival_rate
                    subj_stats[s][2] += time_req
                    subj_stats[s][3] += res_req
                for o in task.get('object_list', []):
                    o_key = o - 50
                    total_obj += 1
                    if o_key not in obj_stats:
                        obj_stats[o_key] = [0, 0.0, 0.0, 0.0]
                    obj_stats[o_key][0] += 1
                    obj_stats[o_key][1] += arrival_rate
                    obj_stats[o_key][2] += time_req
                    obj_stats[o_key][3] += res_req
            subj_metrics = {}
            for s, stats in subj_stats.items():
                count, sum_arr, sum_time, sum_res = stats
                proportion = count / total_subj if total_subj > 0 else 0.0
                avg_arr = sum_arr / count if count > 0 else 0.0
                avg_time = sum_time / count if count > 0 else 0.0
                avg_res = sum_res / count if count > 0 else 0.0
                subj_metrics[s] = [count, proportion, avg_arr, avg_time, avg_res]
            obj_metrics = {}
            for o, stats in obj_stats.items():
                count, sum_arr, sum_time, sum_res = stats
                proportion = count / total_obj if total_obj > 0 else 0.0
                avg_arr = sum_arr / count if count > 0 else 0.0
                avg_time = sum_time / count if count > 0 else 0.0
                avg_res = sum_res / count if count > 0 else 0.0
                obj_metrics[o] = [count, proportion, avg_arr, avg_time, avg_res]
            metrics[uav['id']] = {'subject': subj_metrics, 'object': obj_metrics}
        return metrics

    # ---------- REPLACEMENT ----------
    def compute_uav_metrics_and_similarity(self, total_contents: int = 50):
        """
        Recompute the similarity matrices only when a UAV queue's content changes;
        if the queue is unchanged, read from the cache so results are numerically identical.
        """
        if not self.enable_similarity:
            return {}     # callers that used to expect zeros detect the empty dict

        feature_dim = 5
        metrics = self.compute_uav_content_metrics()     # lightweight statistics
        results = {}

        for uav in self.uavs:
            uav_id = uav['id']
            # (1) Build an order-independent fingerprint of the queue
            q_sig = tuple(sorted(t['task_id'] for t in uav['queue']))

            # (2) Return directly on a cache hit
            c_entry = self._uav_similarity_cache.get(uav_id)
            if c_entry is not None and c_entry['sig'] == q_sig:
                results[uav_id] = c_entry['val']
                continue

            # (3) Otherwise recompute with the original logic
            subj_metrics = metrics[uav_id].get('subject', {})
            obj_metrics  = metrics[uav_id].get('object',  {})
            subject_features = np.zeros((total_contents, feature_dim), dtype=np.float32)
            object_features  = np.zeros((total_contents, feature_dim), dtype=np.float32)

            for i in range(1, total_contents + 1):
                if i in subj_metrics:
                    subject_features[i-1] = np.asarray(subj_metrics[i], dtype=np.float32)
                if i in obj_metrics:
                    object_features[i-1]  = np.asarray(obj_metrics[i], dtype=np.float32)

            val = {
                "subject_features" : subject_features,
                "subject_similarity": cosine_similarity(subject_features),
                "object_features"  : object_features,
                "object_similarity": cosine_similarity(object_features),
            }
            # (4) Update the cache
            self._uav_similarity_cache[uav_id] = {"sig": q_sig, "val": val}
            results[uav_id] = val

        return results
    # ---------- END REPLACEMENT ----------


    def _compute_server_stats_tasks(self, uav, tasks_processed):
        batch_total_triples = sum(t.get('num_triples', 1) for t in tasks_processed) if tasks_processed else 0
        batch_hits = 0
        hit_discount_sum = 0.0
        sim_discount_sum = 0.0
        for t in tasks_processed:
            num_triples = t.get('num_triples', 1)
            # print('num_triples', num_triples)
            # Subject part
            for subj in t.get('subject_list', []):
                if subj in uav['subject_cache']:
                    batch_hits += 1
                    hit_discount_sum += 1.0 / (1 * num_triples)
                elif len(uav['subject_cache']) > 0 and self.enable_similarity:
                    sim_values = [self.subject_similarity_matrix[subj - 1, cache_val - 1]
                                  for cache_val in uav['subject_cache']]
                    max_sim = max(sim_values) if sim_values else 0.0
                    sim_discount_sum += max_sim * 1.5/ (1 * num_triples)
            # Object part
            for obj in t.get('object_list', []):
                if obj in uav['object_cache']:
                    batch_hits += 1
                    hit_discount_sum += 1.0 / (1 * num_triples)
                elif len(uav['object_cache']) > 0 and self.enable_similarity:
                    sim_values = [self.object_similarity_matrix[obj - 51, cache_val - 51]
                                  for cache_val in uav['object_cache']]
                    max_sim = max(sim_values) if sim_values else 0.0
                    sim_discount_sum += max_sim * 1.5/ (1 * num_triples)
        if batch_total_triples > 0:
            hit_rate = batch_hits / (2 * batch_total_triples)
            avg_hit_discount = hit_discount_sum
            avg_sim_discount = sim_discount_sum
            overall_discount = avg_hit_discount + avg_sim_discount
        else:
            hit_rate = 0.0
            avg_hit_discount = 0.0
            avg_sim_discount = 0.0
            overall_discount = 0.0
        if tasks_processed:
            avg_time_req = sum(t['time_req'] for t in tasks_processed) / len(tasks_processed)
            avg_res_req = sum(t['resolution_req'] for t in tasks_processed) / len(tasks_processed)
        else:
            avg_time_req = 0.0
            avg_res_req = 0.0
        print('overall_discount', overall_discount, 'avg_hit_discount', avg_hit_discount, 'avg_sim_discount', avg_sim_discount)
        return {
            'hit_rate': hit_rate,
            'overall_discount': overall_discount,
            'hit_discount': avg_hit_discount,
            'sim_discount': avg_sim_discount,
            'avg_time_req': avg_time_req,
            'avg_res_req': avg_res_req
        }

    # When the server does not cache, return all-zero rewards and skip the caching computation
    def _compute_server_local_reward(self, j, uav, stats_cur, avg_life, num_computed_tasks):
        if not self.enable_server_caching:
            print(f"uav = {j}, no caching: local_reward set to 0")
            return 0.0
        hit_rate = stats_cur['hit_rate']
        discount = stats_cur['hit_discount']
        avg_sim_reuse = stats_cur.get('sim_discount', 0.0)
        if uav['total_tasks_processed'] > 0:
            # local_reward = 1000 * hit_rate + 10 * discount + 10 * avg_sim_reuse
            local_reward =  10 * discount + 10 * avg_sim_reuse
        else:
            local_reward = 0.0
        print('uav =', j,
              'hit_rate =', hit_rate,
              'discount =', discount,
              'avg_sim_reuse =', avg_sim_reuse,
              'local_reward =', local_reward)
        return local_reward

    # old task generation
    # def _generate_tasks_for_user(self, user, image_label_mapping):
    #     e = user['environment']
    #     if np.random.rand() > task_prob[e]:
    #         return
    #     lam = task_rates[e]
    #     num_tasks = np.random.poisson(lam)
    #     if num_tasks <= 0:
    #         return
    #     idx_env = user['environment_index']
    #     resolution_req = resolution_requirement_user[idx_env]
    #     time_req = time_requirement_user[idx_env]
    #     for _ in range(num_tasks):
    #         if idx_env == 0:
    #             image_index = random.randint(1, 10)
    #         elif idx_env == 1:
    #             image_index = random.randint(11, 20)
    #         elif idx_env == 2:
    #             image_index = random.randint(21, 30)
    #         elif idx_env == 3:
    #             image_index = random.randint(31, 40)
    #         else:
    #             image_index = random.randint(41, 50)
    #         triple_info = image_label_mapping.get(image_index, {"subject_indices": [], "object_indices": []})
    #         subject_indices = triple_info.get("subject_indices", [])[:3]  # at most 3 elements
    #         object_indices = triple_info.get("object_indices", [])[:3]  # at most 3 elements
    #         min_len = min(len(subject_indices), len(object_indices))
    #         if min_len == 0:
    #             continue
    #         new_task = {
    #             'task_id': f"user{user['id']}_t{self.global_time}",
    #             'user_id': user['id'],
    #             'time_created': self.global_time,
    #             'image_index': image_index,
    #             'subject_list': subject_indices[:min_len],
    #             'object_list': [obj + 50 for obj in object_indices[:min_len]],
    #             'num_triples': min_len,
    #             'size': TASK_SIZE,
    #             'resolution_req': resolution_req,
    #             'time_req': time_req
    #         }
    #         user['requests'].append(new_task)
    #     print('length of user =', len(user['requests']))

    def zipf_rnd(self, min_val, max_val, s=1.0):
        ranks = np.arange(min_val, max_val+1)
        pmf = 1.0 / (ranks ** s)
        pmf /= pmf.sum()
        cdf = np.cumsum(pmf)
        r = np.random.rand()
        idx = np.searchsorted(cdf, r)
        return ranks[idx]

    def _get_server_obs(self):
        """
        Build the observation (MDP state, Sec. V-A of the paper) of every UAV server:
          First 100 dims: subject request-frequency percentages (50), then object request-frequency percentages (50)
          Last 100 dims: row means (diagonal excluded) of the subject and object similarity matrices, 50 dims each
          The final state is a 200-dim vector
        """
        obs_list = []
        feature_list = []
        uav_sim_data = self.compute_uav_metrics_and_similarity(total_contents=50)
        alpha_weight = 0.5
        beta_weight = 0.5
        for j in range(self.NUM_UAVS):
            subj_ct = [0] * 50
            obj_ct = [0] * 50
            for t in self.uavs[j]['queue']:
                for subj in t.get('subject_list', []):
                    s_idx = subj - 1
                    if 0 <= s_idx < 50:
                        subj_ct[s_idx] += 1
                for obj in t.get('object_list', []):
                    o_idx = obj - 51
                    if 0 <= o_idx < 50:
                        obj_ct[o_idx] += 1
            sum_s = sum(subj_ct)
            sum_o = sum(obj_ct)
            freq_part = []
            if sum_s > 0:
                freq_part.extend([c * 100 / sum_s for c in subj_ct])
            else:
                freq_part.extend([0.0] * 50)
            if sum_o > 0:
                freq_part.extend([c * 100 / sum_o for c in obj_ct])
            else:
                freq_part.extend([0.0] * 50)
            uav_id = self.uavs[j]['id']
            sim_data = uav_sim_data.get(uav_id, {})
            computed_subj_sim = sim_data.get('subject_similarity', np.zeros((50, 50)))
            computed_obj_sim = sim_data.get('object_similarity', np.zeros((50, 50)))
            if not self.enable_similarity:
                combined_subj_sim = np.zeros_like(self.subject_similarity_matrix)
                combined_obj_sim = np.zeros_like(self.object_similarity_matrix)
            else:
                combined_subj_sim = alpha_weight * self.subject_similarity_matrix + beta_weight * computed_subj_sim
                combined_obj_sim = alpha_weight * self.object_similarity_matrix + beta_weight * computed_obj_sim
            n = combined_subj_sim.shape[0]
            subj_avg_vec = (combined_subj_sim.sum(axis=1) - np.diag(combined_subj_sim)) / (n - 1)
            obj_avg_vec = (combined_obj_sim.sum(axis=1) - np.diag(combined_obj_sim)) / (n - 1)
            full_obs = np.concatenate([np.array(freq_part, dtype=np.float32),
                                       subj_avg_vec.astype(np.float32),
                                       obj_avg_vec.astype(np.float32)])
            # -- 5. Extra features: queue length, resolution, CPU frequency -- #
            queue_length = len(self.uavs[j]['queue'])
            # The resolution field stores the total pixel count (e.g. 1280*720); normalize it to [0,1]
            # If uavs[j]['resolution'] is the pixel count:
            resolution = float(self.uavs[j]['resolution']) / (1280.0 * 720.0)

            # cpu_freq is in Hz; normalize to GHz
            frequency = float(self.uavs[j]['cpu_freq']) / 1e9

            # Stack the three items into a (3,) vector and append it to full_obs, giving a (203,) vector
            # extra_feats = np.array([queue_length, resolution, frequency], dtype=np.float32)

            extra_feats = np.array([resolution, frequency], dtype=np.float32)

            feature_vec = np.concatenate([full_obs, extra_feats], axis=0)  # shape = (200+3=203,)

            # feature_vec = extra_feats  # shape = (200+2=202,)

            feature_list.append(feature_vec)

            obs_list.append(full_obs)

        return obs_list

    def _server_update_queues_with_partial(self, act_100):
        if not self.enable_server_caching:
            return [], [0.0] * self.NUM_UAVS, 0.0, {}
        newly_finished = []
        server_local_r_list = [0.0] * self.NUM_UAVS
        partial_rews_dict = {}
        alpha_partial_in_reward = 0.01
        alpha_weight = 0.5
        beta_weight = 0.5
        uav_sim_data = self.compute_uav_metrics_and_similarity(total_contents=50)

        # Inner function that processes a single server, with exception handling
        def process_server(j):
            try:
                partial_vec = np.zeros(100, dtype=np.float32)
                uav_j = self.uavs[j]
                tasks_in_queue_before = len(uav_j['queue'])
                tasks_in_down_before = len(uav_j['downlink_queue'])
                freq = uav_j['cpu_freq']
                c_bits = freq * self.BIT_PER_CYCLE * self.time_slot
                tasks_to_move = []
                remain_bits = c_bits
                uav_id = uav_j['id']
                sim_data = uav_sim_data.get(uav_id, {})
                computed_subj_sim = sim_data.get('subject_similarity', np.zeros((50, 50)))
                computed_obj_sim = sim_data.get('object_similarity', np.zeros((50, 50)))
                if not self.enable_similarity:
                    combined_subj_sim = np.zeros_like(self.subject_similarity_matrix)
                    combined_obj_sim = np.zeros_like(self.object_similarity_matrix)
                else:
                    combined_subj_sim = alpha_weight * self.subject_similarity_matrix + beta_weight * computed_subj_sim
                    combined_obj_sim = alpha_weight * self.object_similarity_matrix + beta_weight * computed_obj_sim
                if tasks_in_queue_before > 0:
                    for t in uav_j['queue'][:]:
                        base_bits = uav_j['resolution'] * self.mu_compute
                        num_triples = t.get('num_triples', 1)
                        total_disc = 0.0

                        # ---------- Subject part (vectorized) ----------
                        subj_ids = np.asarray(t.get('subject_list', []), dtype=np.int32)
                        cache_subj = np.asarray(uav_j['subject_cache'], dtype=np.int32)

                        if subj_ids.size:
                            hit_mask = np.isin(subj_ids, cache_subj)
                            hit_ids = subj_ids[hit_mask]
                            if hit_ids.size:
                                # Cache hit: add 1/num_triples to the matching cached contents in one shot
                                contrib = 1.0 / num_triples
                                np.add.at(partial_vec, hit_ids - 1, contrib)
                                total_disc += hit_ids.size * contrib

                            miss_ids = subj_ids[~hit_mask]
                            if self.enable_similarity and miss_ids.size and cache_subj.size:
                                sim_blk = combined_subj_sim[np.ix_(miss_ids - 1, cache_subj - 1)]
                                max_idx = sim_blk.argmax(axis=1)
                                max_sim = sim_blk[np.arange(miss_ids.size), max_idx]
                                best_cached = cache_subj[max_idx]

                                contrib_arr = max_sim * 1.5 / num_triples
                                np.add.at(partial_vec, best_cached - 1, contrib_arr)
                                total_disc += contrib_arr.sum()

                        # ---------- Object part (vectorized, mind the index offset) ----------
                        obj_ids = np.asarray(t.get('object_list', []), dtype=np.int32)
                        cache_obj = np.asarray(uav_j['object_cache'], dtype=np.int32)

                        if obj_ids.size:
                            hit_mask = np.isin(obj_ids, cache_obj)
                            hit_ids = obj_ids[hit_mask]
                            if hit_ids.size:
                                contrib = 1.0 / num_triples
                                np.add.at(partial_vec, 50 + hit_ids - 51, contrib)
                                total_disc += hit_ids.size * contrib

                            miss_ids = obj_ids[~hit_mask]
                            if self.enable_similarity and miss_ids.size and cache_obj.size:
                                sim_blk = combined_obj_sim[np.ix_(miss_ids - 51, cache_obj - 51)]
                                max_idx = sim_blk.argmax(axis=1)
                                max_sim = sim_blk[np.arange(miss_ids.size), max_idx]
                                best_cached = cache_obj[max_idx]

                                contrib_arr = max_sim * 1.5 / num_triples
                                np.add.at(partial_vec, 50 + best_cached - 51, contrib_arr)
                                total_disc += contrib_arr.sum()

                        total_discount = total_disc
                        factor = 1.0 - total_discount
                        if factor < 0:
                            factor = 0
                        needed = base_bits * factor
                        if needed <= remain_bits:
                            uav_j['total_tasks_processed'] += 1
                            for subj in t.get('subject_list', []):
                                if subj in uav_j['subject_cache']:
                                    uav_j['cache_hits'] += 1
                            for obj in t.get('object_list', []):
                                if obj in uav_j['object_cache']:
                                    uav_j['cache_hits'] += 1
                            uav_j['cache_discount_acc'] += total_discount
                            tasks_to_move.append(t)
                            remain_bits -= needed
                        else:
                            break
                    for tsk in tasks_to_move:
                        uav_j['queue'].remove(tsk)
                        tsk.setdefault('remaining_down_bits', uav_j['resolution'])
                        uav_j['downlink_queue'].append(tsk)
                # Downlink queue processing and statistics keep the original logic
                from collections import defaultdict
                old_down = uav_j['downlink_queue'][:]
                uav_j['downlink_queue'].clear()
                user_map = defaultdict(list)
                for tk in old_down:
                    user_map[tk['user_id']].append(tk)
                newly_finished_local = []
                for uid, tklist in user_map.items():
                    user_obj = self.users[uid - 1]
                    d_rate = self._calculate_downlink_rate(uav_j, user_obj)
                    capacity_bits = d_rate * self.time_slot
                    new_tlist = []
                    for tk2 in tklist:
                        if tk2['remaining_down_bits'] <= capacity_bits:
                            capacity_bits -= tk2['remaining_down_bits']
                            tk2['remaining_down_bits'] = 0
                            user_obj['received'].append(tk2)
                            newly_finished_local.append((uav_j, tk2))
                        else:
                            tk2['remaining_down_bits'] -= capacity_bits
                            capacity_bits = 0
                            new_tlist.append(tk2)
                    for tkx in new_tlist:
                        uav_j['downlink_queue'].append(tkx)
                stats_cur = self._compute_server_stats_tasks(uav_j, tasks_to_move)
                local_r = 0.0
                if (len(uav_j['queue']) > 0 or len(uav_j['downlink_queue']) > 0 or len(tasks_to_move) > 0):
                    local_r = self._compute_server_local_reward(j, uav_j, stats_cur, 0.0, len(tasks_to_move))
                effective_partial = partial_vec * act_100[j]
                shaping_sum = effective_partial.sum()
                local_r += alpha_partial_in_reward * shaping_sum
                local_r /= 0.1
                stats_cur['local_reward'] = local_r
                uav_j['stats_history'].append(stats_cur)
                return j, local_r, partial_vec, newly_finished_local
            except Exception as e:
                print(f"Error in process_server for server {j}: {e}")
                raise

        newly_finished_all = []
        # with ThreadPoolExecutor(max_workers=NUM_UAVS) as executor:
        #     futures = [executor.submit(process_server, j) for j in range(NUM_UAVS)]
        #     for future in futures:
        #         try:
        #             j, local_r, partial_vec, newly_finished_local = future.result()
        #             server_local_r_list[j] = local_r
        #             partial_rews_dict[j] = partial_vec
        #             newly_finished_all.extend(newly_finished_local)
        #         except Exception as e:
        #             print("Exception in server processing:", e)

        for j in range(self.NUM_UAVS):  # sequential processing is sufficient
            try:
                j, local_r, partial_vec, newly_finished_local = process_server(j)
                server_local_r_list[j] = local_r
                partial_rews_dict[j] = partial_vec
                newly_finished_all.extend(newly_finished_local)
            except Exception as e:
                print("Exception in server processing:", e)

        nonzero_r = [r for r in server_local_r_list if abs(r) > 1e-9]
        server_global_r = float(np.mean(nonzero_r)) if nonzero_r else 0.0
        return newly_finished_all, server_local_r_list, server_global_r, partial_rews_dict

    def _server_update_queues_no_cache(self):
        """
        Without caching, process tasks at each server's full computing capacity and count the tasks finished in this slot.
        No cache matching or similarity computation is performed; every task's computing demand is used directly:
            required_bits = resolution * mu_compute
        The downlink transmission is also computed at the standard downlink rate.
        Returns: (newly_finished, server_local_r_list, global_r, partial_rews_dict)
        where:
            newly_finished: tasks whose downlink finished in this slot, as (uav, task) tuples
            server_local_r_list: per-server local rewards (0 without caching)
            global_r: global reward computed from the delay of finished tasks
            partial_rews_dict: empty dict (no partial rewards without caching)
        """
        newly_finished = []
        server_local_r_list = [0.0] * self.NUM_UAVS

        # Process tasks on every server (UAV)
        for j in range(self.NUM_UAVS):
            uav = self.uavs[j]
            # Bits the server can compute in this slot
            freq = uav['cpu_freq']
            c_bits = freq * self.BIT_PER_CYCLE * self.time_slot
            tasks_to_move = []
            remain_bits = c_bits

            # Scan the compute queue without any discount and check whether each task can be finished
            for task in uav['queue'][:]:
                required_bits = uav['resolution'] * self.mu_compute
                if required_bits <= remain_bits:
                    tasks_to_move.append(task)
                    remain_bits -= required_bits
                else:
                    break

            # Move finished tasks from the compute queue to the downlink queue and init the remaining downlink bits
            for task in tasks_to_move:
                uav['queue'].remove(task)
                task.setdefault('remaining_down_bits', uav['resolution'])
                uav['downlink_queue'].append(task)

            # Process the downlink queue
            from collections import defaultdict
            old_down = uav['downlink_queue'][:]
            uav['downlink_queue'].clear()
            user_map = defaultdict(list)
            for task in old_down:
                user_map[task['user_id']].append(task)
            for uid, tasks in user_map.items():
                user = self.users[uid - 1]
                d_rate = self._calculate_downlink_rate(uav, user)
                capacity_bits = d_rate * self.time_slot
                remaining_tasks = []
                for task in tasks:
                    if task['remaining_down_bits'] <= capacity_bits:
                        capacity_bits -= task['remaining_down_bits']
                        task['remaining_down_bits'] = 0
                        user['received'].append(task)
                        newly_finished.append((uav, task))
                    else:
                        task['remaining_down_bits'] -= capacity_bits
                        capacity_bits = 0
                        remaining_tasks.append(task)
                for task in remaining_tasks:
                    uav['downlink_queue'].append(task)
            # Local reward is 0 without caching
            server_local_r_list[j] = 0.0

        # Global reward based on the delay of every newly finished task
        if len(newly_finished) > 0:
            step_rewards = []
            for (uav_obj, task) in newly_finished:
                user = self.users[task['user_id'] - 1]
                freq_j = uav_obj['cpu_freq']
                sr_res = uav_obj['resolution']
                # Fixed computing demand: required bits
                base_bits = sr_res * self.mu_compute
                if freq_j * self.BIT_PER_CYCLE > 1e-9:
                    compute_time = base_bits / (freq_j * self.BIT_PER_CYCLE)
                else:
                    compute_time = 999999.9
                total_down_bits = sr_res * self.mu_downlink
                d_rate = self._calculate_downlink_rate(uav_obj, user)
                eff_rate = d_rate * self.COMPRESSION_RATE
                if eff_rate > 1e-9:
                    down_time = total_down_bits / eff_rate
                else:
                    down_time = 999999.9
                T_est = compute_time + down_time
                # Delay = time from task creation to completion (plus 1 slot as compensation), as in the original logic
                delay = (self.global_time - task['time_created'] + T_est) * self.time_slot
                if delay < 1e-9:
                    time_reward = 0.0
                else:
                    time_reward = (task['time_req'] - delay)
                res_ratio = sr_res / task['resolution_req']
                r = 0.8 * time_reward + 0.2 * res_ratio
                step_rewards.append(r)
        #     global_r = float(np.mean(step_rewards)) * 1000
        # else:
            global_r = 0.0

        return newly_finished, server_local_r_list, global_r, {}


    def _compute_user_local_reward(self, user_i, chosen_uav):
        tasks_list = self.v_queue_tasks[chosen_uav][user_i]
        if len(tasks_list) == 0:
            return 0.0
        user = self.users[user_i]
        uav = self.uavs[chosen_uav]
        freq_j = uav['cpu_freq']
        sr_res = uav['resolution']
        cache_subj = set(uav['subject_cache'])
        cache_obj = set(uav['object_cache'])
        uav_sim_data = self.compute_uav_metrics_and_similarity(total_contents=50)
        uav_id = uav['id']
        sim_data = uav_sim_data.get(uav_id, {})
        computed_subj_sim = sim_data.get('subject_similarity', np.zeros((50, 50)))
        computed_obj_sim = sim_data.get('object_similarity', np.zeros((50, 50)))
        alpha_weight = 0.5
        beta_weight = 0.5
        if not self.enable_similarity:
            combined_subj_sim = np.zeros_like(self.subject_similarity_matrix)
            combined_obj_sim = np.zeros_like(self.object_similarity_matrix)
        else:
            combined_subj_sim = alpha_weight * self.subject_similarity_matrix + beta_weight * computed_subj_sim
            combined_obj_sim = alpha_weight * self.object_similarity_matrix + beta_weight * computed_obj_sim
        total_compute_bits = 0.0
        for tsk in tasks_list:
            base_bits = sr_res * self.mu_compute
            num_triples = tsk.get('num_triples', 1)
            total_disc = 0.0
            for subj in tsk.get('subject_list', []):
                if subj in cache_subj:
                    disc = 1.0 / (1 * num_triples)
                elif len(cache_subj) > 0 and self.enable_similarity:
                    sim_vals = [combined_subj_sim[subj - 1, c - 1] for c in cache_subj]
                    max_sim = max(sim_vals) if sim_vals else 0.0
                    disc = max_sim * 1.5/ (1 * num_triples)
                else:
                    disc = 0.0
                total_disc += disc
            for obj in tsk.get('object_list', []):
                if obj in cache_obj:
                    disc = 1.0 / (1 * num_triples)
                elif len(cache_obj) > 0 and self.enable_similarity:
                    sim_vals = [combined_obj_sim[obj - 51, c - 51] for c in cache_obj]
                    max_sim = max(sim_vals) if sim_vals else 0.0
                    disc = max_sim * 1.5/ (1 * num_triples)
                else:
                    disc = 0.0
                total_disc += disc
            total_discount = total_disc
            factor = 1.0 - total_discount
            if factor < 0.0:
                factor = 0.0
            needed = base_bits * factor
            total_compute_bits += needed
        if freq_j * self.BIT_PER_CYCLE > 1e-9:
            compute_time = total_compute_bits / (freq_j * self.BIT_PER_CYCLE)
        else:
            compute_time = 999999.9
        total_down_bits = len(tasks_list) * (sr_res * self.mu_downlink)
        d_rate = self._calculate_downlink_rate(uav, user)
        eff_rate = d_rate * self.COMPRESSION_RATE
        if eff_rate > 1e-9:
            down_time = total_down_bits / eff_rate
        else:
            down_time = 999999.9
        T_est = (compute_time + down_time)
        last_task = tasks_list[-1]
        t_req = last_task['time_req']
        r_req = last_task['resolution_req']
        time_part = t_req - T_est
        res_part = sr_res / r_req
        local_r = 0.95 * time_part + 0.05 * res_part
        return local_r



    def _calculate_upload_rate(self, user, uav):
        a, b, y0, y1 = self.channel_params[user['environment']]
        bw = self.bandwidth_map_user[user['environment']]
        up = np.array(user['position'])
        vp = np.array(uav['position'])
        dist3 = np.linalg.norm(up - vp)
        dist2 = np.linalg.norm(up[:2] - vp[:2])
        if dist2 == 0:
            psi = pi / 2
        else:
            psi = np.arctan(self.UAV_HEIGHT / dist2)
        p0 = 1 / (1 + a * np.exp(-b * (psi - a)))
        fspl = ((4 * pi * self.CARRIER_FREQ * dist3) / 3e8) ** 2
        L = p0 * fspl * (10 ** (y0 / 20)) + (1 - p0) * fspl * (10 ** (y1 / 20))
        snr = (self.P_USER * self.G_USER) / (L * self.NOISE_POWER * bw)
        return bw * log2(1 + snr)

    def _calculate_downlink_rate(self, uav, user):
        a, b, y0, y1 = self.channel_params[user['environment']]
        bandwidth = uav['bandwidth']
        upos = np.array(user['position'])
        vpos = np.array(uav['position'])
        dist_3d = np.linalg.norm(vpos - upos)
        dist_2d = np.linalg.norm(vpos[:2] - upos[:2])
        if dist_2d == 0:
            psi = pi / 2
        else:
            psi = np.arctan(self.UAV_HEIGHT / dist_2d)
        p0 = 1 / (1 + a * exp(-b * (psi - a)))
        fspl = ((4 * pi * self.CARRIER_FREQ * dist_3d) / 3e8) ** 2
        L = p0 * fspl * (10 ** (y0 / 20)) + (1 - p0) * fspl * (10 ** (y1 / 20))
        snr = (self.P_UAV * self.G_UAV) / (L * self.NOISE_POWER * bandwidth)
        return bandwidth * log2(1 + snr)

    def _compute_uploadable_tasks(self, rate_bps):
        cap = int(floor(rate_bps * self.time_slot / self.TASK_SIZE))
        return max(cap, 0)

    def assign_users_to_closest_uav(self, users, uavs):
        """
        Bind every user to the nearest UAV server that offers the same rendering rate.
        Return a vector holding the server index chosen by each user.
        """
        user_assignments = []

        for user in users:
            user_pos = np.array(user['position'][:2])
            user_res = user['resolution_req']
            min_dist = float('inf')
            chosen_server_idx = -1

            for idx, uav in enumerate(uavs):
                if uav['resolution'] == user_res:
                    uav_pos = np.array(uav['position'][:2])
                    dist = np.linalg.norm(user_pos - uav_pos)
                    if dist < min_dist:
                        min_dist = dist
                        chosen_server_idx = idx

            user_assignments.append(chosen_server_idx)

        print(f"[ASSIGN] User assignments to UAV servers completed.")
        return user_assignments

    def compute_features_for_tasks(self, uav_id, tasks):
        """
        Compute the task feature vectors of the `tasks` just uploaded to UAV `uav_id`.
        Returns List[np.ndarray]; see the final concatenation below for the vector length.
        """
        total_labels = 50
        # 1) Row means of the fused similarity matrices
        sim_data = self.compute_uav_metrics_and_similarity(total_contents=total_labels).get(uav_id, {})
        subj_sim = sim_data.get('subject_similarity', np.zeros((total_labels, total_labels)))
        obj_sim = sim_data.get('object_similarity', np.zeros((total_labels, total_labels)))
        if self.enable_similarity:
            alpha, beta = 0.5, 0.5
            subj_comb = alpha * self.subject_similarity_matrix + beta * subj_sim
            obj_comb = alpha * self.object_similarity_matrix + beta * obj_sim
        else:
            subj_comb = np.zeros_like(self.subject_similarity_matrix)
            obj_comb = np.zeros_like(self.object_similarity_matrix)
        n = total_labels
        subj_avg = (subj_comb.sum(axis=1) - np.diag(subj_comb)) / (n - 1)
        obj_avg = (obj_comb.sum(axis=1) - np.diag(obj_comb)) / (n - 1)

        # 2) Subject/object request counts of this batch
        subj_ct = [0] * total_labels
        obj_ct = [0] * total_labels
        for t in tasks:
            for s in t.get('subject_list', []):
                idx = s - 1
                if 0 <= idx < total_labels:
                    subj_ct[idx] += 1
            for o in t.get('object_list', []):
                idx = o - 51
                if 0 <= idx < total_labels:
                    obj_ct[idx] += 1
        sum_s, sum_o = sum(subj_ct), sum(obj_ct)
        freq = (
                   [c * 100 / sum_s for c in subj_ct] if sum_s > 0 else [0.0] * total_labels
               ) + (
                   [c * 100 / sum_o for c in obj_ct] if sum_o > 0 else [0.0] * total_labels
               )

        # 3) Extra UAV parameters: resolution & cpu_freq
        uav = self.uavs[uav_id]
        resolution = float(uav['resolution']) / (1280.0 * 720.0)
        frequency = float(uav['cpu_freq']) / 1e9

        # (Optional) append this resolution/CPU to the history lists for later standardization
        self.historical_resolutions.append(resolution)
        self.historical_cpu_freqs.append(frequency)

        # 4) Concatenate into a single feature vector
        vec = np.concatenate([
            np.array(freq, dtype=np.float32),
            subj_avg.astype(np.float32),
            obj_avg.astype(np.float32),
            np.array([resolution, frequency], dtype=np.float32)
        ], axis=0)

        # # -- call the preprocessing function --
        # # hist_res and hist_cpu (historical distributions) must be prepared beforehand
        # vec = preprocess_feature_vector(
        #         freq_part = vec[:100].tolist(),
        #     subj_avg = vec[100:150],
        #     obj_avg = vec[150:200],
        #     resolution = vec[-2],  # the last two dims are resolution & cpu
        #     cpu_freq = vec[-1],
        #     hist_res = self.historical_resolutions,
        #     hist_cpu = self.historical_cpu_freqs
        #                         )


        return [vec]  # return a list so that multiple vectors can be added later

    def step(self, server_agent):
        print(f"[DEBUG] before step: current_step_in_episode={self.current_step_in_episode}")

        if self.current_step_in_episode >= self.episode_length:
            return self.get_global_obs(), True, {
                "info_user": {},
                "info_server": {},
                "global_r": 0.0,
                "done": True
            }

        print(f"\n[STEP] time={self.global_time}, step_in_ep={self.current_step_in_episode + 1}")

        # 1. Generate tasks for each user
        for i in range(self.num_agents):
            self._generate_tasks_for_user(self.users[i], self.image_label_mapping)
            self.v_user_requests[i] = len(self.users[i]['requests'])

        # Record initial queue lengths
        if not hasattr(self, 'server_queue_history'):
            self.server_queue_history = []
        current_queue_lengths = [len(uav['queue']) + len(uav['downlink_queue']) for uav in self.uavs]
        self.server_queue_history.append(current_queue_lengths)

        # Initialize user decision recording
        user_actions = [-1] * self.num_agents
        user_logps = [0.0] * self.num_agents
        user_values = [0.0] * self.num_agents
        user_locals = [0.0] * self.num_agents

        obs_list = self.get_global_obs()

        user_actions = self.assign_users_to_closest_uav(self.users, self.uavs)

        self.latest_transmitted_features = {j: [] for j in range(self.NUM_UAVS)}

        for user_i, act in enumerate(user_actions):
            if act == -1 or self.v_user_requests[user_i] == 0:
                continue

            upload_rate = self._calculate_upload_rate(self.users[user_i], self.uavs[act])
            capacity = self._compute_uploadable_tasks(rate_bps=upload_rate)
            n_tasks = min(self.v_user_requests[user_i], capacity)

            tasks_to_upload = self.users[user_i]['requests'][:n_tasks]
            self.users[user_i]['requests'] = self.users[user_i]['requests'][n_tasks:]

            # Update the virtual queues
            self.v_queue_tasks[act][user_i] += tasks_to_upload
            self.v_uav_downlink_user[act][user_i] += n_tasks

            for task in tasks_to_upload:
                task['server_resolution'] = self.uavs[act]['resolution']
                self.uavs[act]['queue'].append(task)

            # -- Compute the features of the tasks uploaded in this step -- #
            new_feats = self.compute_features_for_tasks(self.uavs[act]['id'] - 1, tasks_to_upload)
            # Store as an attribute for later access/logging
            self.latest_transmitted_features[act] = new_feats

        server_feature_list  = self.latest_transmitted_features

        for j in range(self.NUM_UAVS):
            self.uavs[j]['step_rewards'] = []

        # 4. Server side follows the original logic
        server_obs_list = None
        # server_feature_list = None
        server_actions_100 = []
        server_logps_10 = []
        server_values_10 = []
        server_local_r_list = []
        server_global_r = 0.0
        if self.enable_server_caching and (server_agent is not None):
            server_obs_list = self._get_server_obs()
            serv_acts, serv_logp_s, serv_val_s = server_agent.get_actions_all_servers(server_obs_list)
            for j in range(self.NUM_UAVS):
                self.uavs[j]['prev_subject_cache'] = self.uavs[j]['subject_cache'][:]
                self.uavs[j]['prev_object_cache'] = self.uavs[j]['object_cache'][:]
            for j in range(self.NUM_UAVS):
                act_100 = serv_acts[j]
                lp_j = serv_logp_s[j]
                val_j = serv_val_s[j]
                subj_new, obj_new = server_agent.decode_action(act_100)
                self.uavs[j]['subject_cache'] = subj_new
                self.uavs[j]['object_cache'] = obj_new
                server_actions_100.append(act_100)
                server_logps_10.append(lp_j)
                server_values_10.append(val_j)
                print('server =', j, 'subject cache =', self.uavs[j]['subject_cache'],
                      'object cache =', self.uavs[j]['object_cache'])
        # for user_i in range(self.num_agents):
        #     if user_actions[user_i] == -1:
        #         user_locals[user_i] = 0.0
        #     else:
        #         local_r_i = self._compute_user_local_reward(user_i, user_actions[user_i])
        #         user_locals[user_i] = local_r_i * 200

        if not self.enable_server_caching:
            newly_finished, server_local_r_list, server_global_r, partial_rews_dict = self._server_update_queues_no_cache()
        else:
            newly_finished, server_local_r_list, server_global_r, partial_rews_dict = self._server_update_queues_with_partial(
                server_actions_100)

        if self.enable_server_caching and (server_agent is not None):
            alpha_s = server_agent.alpha
            beta_s = server_agent.beta
        else:
            alpha_s = beta_s = 0.0

        # Global reward (original logic)
        if len(newly_finished) > 0:
            step_rewards = []
            for (uav_obj, task) in newly_finished:
                user = self.users[task['user_id'] - 1]
                freq_j = uav_obj['cpu_freq']
                sr_res = uav_obj['resolution']
                cache_subj = set(uav_obj['subject_cache'])
                cache_obj = set(uav_obj['object_cache'])
                uav_sim_data = self.compute_uav_metrics_and_similarity(total_contents=50)
                uav_id = uav_obj['id']
                sim_data = uav_sim_data.get(uav_id, {})
                computed_subj_sim = sim_data.get('subject_similarity', np.zeros((50, 50)))
                computed_obj_sim = sim_data.get('object_similarity', np.zeros((50, 50)))
                alpha_weight = 0.5
                beta_weight = 0.5
                if not self.enable_similarity:
                    combined_subj_sim = np.zeros_like(self.subject_similarity_matrix)
                    combined_obj_sim = np.zeros_like(self.object_similarity_matrix)
                else:
                    combined_subj_sim = alpha_weight * self.subject_similarity_matrix + beta_weight * computed_subj_sim
                    combined_obj_sim = alpha_weight * self.object_similarity_matrix + beta_weight * computed_obj_sim
                base_bits = sr_res * self.mu_compute
                num_triples = task.get('num_triples', 1)
                total_disc = 0.0
                for subj in task.get('subject_list', []):
                    if subj in cache_subj:
                        disc = 1.0 / (1 * num_triples)
                    elif len(cache_subj) > 0 and self.enable_similarity:
                        sim_vals = [combined_subj_sim[subj - 1, c - 1] for c in cache_subj]
                        max_sim = max(sim_vals) if sim_vals else 0.0
                        disc = max_sim * 1.5/ (1 * num_triples)
                    else:
                        disc = 0.0
                    total_disc += disc
                for obj in task.get('object_list', []):
                    if obj in cache_obj:
                        disc = 1.0 / (1 * num_triples)
                    elif len(cache_obj) > 0 and self.enable_similarity:
                        sim_vals = [combined_obj_sim[obj - 51, c - 51] for c in cache_obj]
                        max_sim = max(sim_vals) if sim_vals else 0.0
                        disc = max_sim * 1.5/ (1 * num_triples)
                    else:
                        disc = 0.0
                    total_disc += disc
                total_discount = total_disc
                factor = 1.0 - total_discount
                if factor < 0.0:
                    factor = 0.0
                needed = base_bits * factor
                if freq_j * self.BIT_PER_CYCLE > 1e-9:
                    compute_time = needed / (freq_j * self.BIT_PER_CYCLE)
                else:
                    compute_time = 999999.9
                total_down_bits = sr_res * self.mu_downlink
                d_rate = self._calculate_downlink_rate(uav_obj, user)
                eff_rate = d_rate * self.COMPRESSION_RATE
                if eff_rate > 1e-9:
                    down_time = total_down_bits / eff_rate
                else:
                    down_time = 999999.9
                T_est = compute_time + down_time
                delay = (self.global_time - task['time_created'] + T_est) * self.time_slot
                if delay < 1e-9:
                    time_reward = 0.0
                else:
                    # time_reward = (task['time_req'] / (delay + 0.5))
                    if delay <= task['time_req']:
                        time_reward = 1
                    else:
                        time_reward = task['time_req'] / delay

                if sr_res >= task['resolution_req']:
                    res_ratio = 1
                else:
                    res_ratio = sr_res / task['resolution_req']

                r = (time_reward) * 100
                step_rewards.append(r)
                # Add the reward to the corresponding server
                self.uavs[uav_id-1]['step_rewards'].append(r)

            global_r = float(np.mean(step_rewards)) * 1000
        else:
            global_r = 0.0
        print('user_global_r =', global_r)
        user_combine = 0
        self.current_step_in_episode += 1
        self.global_time += 1
        done = (self.current_step_in_episode >= self.episode_length)

        # 3. Per-server reward bookkeeping
        server_task_rewards = []
        for j in range(self.NUM_UAVS):
            if len(self.uavs[j]['step_rewards']) > 0:
                server_local_reward = np.mean(self.uavs[j]['step_rewards']) * 100 #100
            else:
                server_local_reward = 0.0
            server_task_rewards.append(server_local_reward)

        # Update the server-side combined reward (utility U_j, Eq. 16)
        server_combined_r_list = []
        for j in range(self.NUM_UAVS):
            c_rj = alpha_s * server_local_r_list[j] + beta_s * server_task_rewards[j]  # note: server_task_rewards
            server_combined_r_list.append(c_rj)

        info_user = {
            "obs_list": obs_list,
            "actions": user_actions,
            "logps": user_logps,
            "values": user_values,
            "local_r": user_locals,
            "combined_r": user_combine
        }
        if not self.enable_server_caching or (server_agent is None):
            info_server = {
                "obs_list": None,
                "actions": [],
                "logps": [],
                "values": [],
                "local_r": [0.0] * self.NUM_UAVS,
                "combined_r": [0.0] * self.NUM_UAVS,
                "global_r": 0.0,
                "partial_rews": {}
            }
        else:
            info_server = {
                "obs_list": server_obs_list,
                "actions": server_actions_100,
                "logps": server_logps_10,
                "values": server_values_10,
                "local_r": server_local_r_list,
                "combined_r": server_combined_r_list,
                "global_r": server_global_r,
                "partial_rews": partial_rews_dict
            }
        info_all = {
            "info_user": info_user,
            "info_server": info_server,
            "global_r": global_r,
            "done": done
        }
        return self.get_global_obs(), done, info_all, server_feature_list

