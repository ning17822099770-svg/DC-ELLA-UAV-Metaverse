"""User-side MAPPO agent (kept for completeness).

Imported by the main scripts but not instantiated in the reported
experiments: the user-server association is part of the server action
(see Server_agent.py).

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

from network_define import user_GRUActorNet, user_GRUCriticNet

class MAPPOAgent:
    def __init__(self, num_agents=25, obs_dim=31, action_dim=10, device='cpu',
                 hidden_dim=128, lr=1e-4, gamma=0.99, lam=0.95,
                 clip_ratio=0.4, entropy_coef=0.1, max_grad_norm=0.5):
        self.device = device
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.actors = []
        self.critics = []
        for _ in range(num_agents):
            actor = user_GRUActorNet(obs_dim, action_dim, hidden_dim).to(self.device)
            critic = user_GRUCriticNet(obs_dim, hidden_dim).to(self.device)
            self.actors.append(actor)
            self.critics.append(critic)
        self.actor_h = [None] * num_agents
        self.critic_h = [None] * num_agents
        params = []
        for i in range(num_agents):
            params += list(self.actors[i].parameters())
            params += list(self.critics[i].parameters())
        self.optimizer = optim.Adam(params, lr=lr)
    def reset_hidden_states(self):
        self.actor_h = [None] * self.num_agents
        self.critic_h = [None] * self.num_agents
    @torch.no_grad()
    def get_actions_single(self, obs_i, agent_i):
        obs_t = torch.tensor(obs_i, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, h_new = self.actors[agent_i](obs_t, self.actor_h[agent_i])
        self.actor_h[agent_i] = h_new
        dist = Categorical(logits=logits)
        act = dist.sample()
        lp = dist.log_prob(act)
        v_out, h_c_new = self.critics[agent_i](obs_t, self.critic_h[agent_i])
        self.critic_h[agent_i] = h_c_new
        return act.item(), lp.item(), v_out.item()
    def compute_logp_value_entropy_single_agent(self, obs_batch, act_batch, agent_i,
                                                 rnn_actor_h=None, rnn_critic_h=None):
        T = obs_batch.size(0)
        logp_list = []
        ent_list = []
        v_list = []
        h_a = rnn_actor_h
        h_c = rnn_critic_h
        for t in range(T):
            obs_t = obs_batch[t].unsqueeze(0)
            act_t = act_batch[t].unsqueeze(0)
            logits, h_a = self.actors[agent_i](obs_t, h_a)

            # Detect and fix invalid logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
                print("[WARNING] logits contain NaN/inf, replaced by stable numerical values.")

            dist = Categorical(logits=logits)
            lp = dist.log_prob(act_t)
            ent = dist.entropy()
            v_o, h_c = self.critics[agent_i](obs_t, h_c)
            logp_list.append(lp[0])
            ent_list.append(ent[0])
            v_list.append(v_o[0,0])
        logp_all = torch.stack(logp_list, dim=0)
        ent_all = torch.stack(ent_list, dim=0)
        v_all = torch.stack(v_list, dim=0)
        return logp_all, ent_all, v_all, h_a, h_c

    def update_single(self, agent_i, single_buf, train_iters=10, batch_size=64):
        obs_np = single_buf["obs"]
        T = obs_np.shape[0]
        if T == 0:
            return 0.0, 0.0, 0.0
        obs_dim = obs_np.shape[2]
        obs_flat = obs_np.reshape(T, obs_dim)
        obs_t = torch.tensor(obs_flat, dtype=torch.float32, device=self.device)
        act_t = torch.tensor(single_buf["actions"], dtype=torch.long, device=self.device).view(T)
        logp_old_t = torch.tensor(single_buf["logp"], dtype=torch.float32, device=self.device).view(T)
        adv_t = torch.tensor(single_buf["adv"], dtype=torch.float32, device=self.device).view(T)
        ret_t = torch.tensor(single_buf["returns"], dtype=torch.float32, device=self.device).view(T)
        with torch.no_grad():
            _, _, v_seq, _, _ = self.compute_logp_value_entropy_single_agent(obs_t, act_t, agent_i, None, None)
            v_old = v_seq
        # adv_m, adv_s = adv_t.mean(), adv_t.std() + 1e-8
        if len(adv_t) <= 1:
            print("[Warning] adv_t too short (length={}), setting adv_m=0, adv_s=1.0".format(len(adv_t)))
            adv_m = torch.tensor(0.0, device=adv_t.device)
            adv_s = torch.tensor(1.0, device=adv_t.device)
        else:
            adv_m, adv_s = adv_t.mean(), adv_t.std()
            adv_s = torch.clamp(adv_s, min=1e-4)  # avoid a too-small standard deviation

        adv_t = (adv_t - adv_m) / adv_s
        idxes = np.arange(T)
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_loss = 0.0
        total_updates = 0
        for _ in range(train_iters):
            np.random.shuffle(idxes)
            start = 0
            while start < T:
                end = start + batch_size
                mb = idxes[start:end]
                start = end
                total_updates += 1
                obs_b = obs_t[mb]
                act_b = act_t[mb]
                logp_old_b = logp_old_t[mb]
                adv_b = adv_t[mb]
                ret_b = ret_t[mb]
                v_old_b = v_old[mb]
                logp_new_b, ent_new_b, v_new_b, _, _ = self.compute_logp_value_entropy_single_agent(obs_b, act_b, agent_i, None, None)
                ratio = torch.exp(logp_new_b - logp_old_b)
                unclipped = ratio * adv_b
                clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_b
                loss_actor = - torch.mean(torch.min(unclipped, clipped))
                clip_v = 0.1
                v_clipped = v_old_b + torch.clamp(v_new_b - v_old_b, -clip_v, clip_v)
                c_unclip = (ret_b - v_new_b)**2
                c_clip = (ret_b - v_clipped)**2
                loss_critic = torch.mean(torch.max(c_unclip, c_clip))
                ent = torch.mean(ent_new_b)
                loss_entropy = - self.entropy_coef * ent
                loss = loss_actor + 0.5 * loss_critic + loss_entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total_actor_loss += loss_actor.item()
                total_critic_loss += loss_critic.item()
                total_loss += loss.item()
        avg_actor = total_actor_loss / total_updates
        avg_critic = total_critic_loss / total_updates
        avg_loss = total_loss / total_updates
        return avg_actor, avg_critic, avg_loss
    def parameters(self):
        for i in range(self.num_agents):
            yield from self.actors[i].parameters()
            yield from self.critics[i].parameters()