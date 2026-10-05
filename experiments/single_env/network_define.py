"""Actor / critic networks.

ServerActorNetNoGRUForELLA is the server PPO actor used in the paper
(obs -> 32 -> 32 -> action, Sec. VI). extract_policy_parameters() flattens the
weights and bias of the 32x32 `policy_mlp` layer into theta, and
load_policy_parameters(theta) writes a DC-ELLA reconstruction theta = L s_m
back into that layer. The remaining layers (obs_norm, input_mlp, fc_out) are
saved separately as server_actor_<j>_non_policy_params.pt.

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



##################################################
# ========== User Actor/Critic Net ==========
##################################################
class user_GRUActorNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128, use_gru=True):
        super().__init__()
        self.use_gru = use_gru
        self.mlp = nn.Linear(obs_dim, hidden_dim)
        if self.use_gru:
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, action_dim)
    def forward(self, obs, h_in=None):
        x = torch.relu(self.mlp(obs))
        if self.use_gru:
            x = x.unsqueeze(1)
            if h_in is None:
                h_in = torch.zeros(1, obs.size(0), x.size(-1), device=obs.device)
            x, h_out = self.gru(x, h_in)
            x = x.squeeze(1)
        else:
            h_out = h_in
        logits = self.fc_out(x)
        return logits, h_out

class user_GRUCriticNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim=128, use_gru=True):
        super().__init__()
        self.use_gru = use_gru
        self.mlp = nn.Linear(obs_dim, hidden_dim)
        if self.use_gru:
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)
    def forward(self, obs, h_in=None):
        x = torch.relu(self.mlp(obs))
        if self.use_gru:
            x = x.unsqueeze(1)
            if h_in is None:
                h_in = torch.zeros(1, obs.size(0), x.size(-1), device=obs.device)
            x, h_out = self.gru(x, h_in)
            x = x.squeeze(1)
        else:
            h_out = h_in
        v = self.fc_out(x)
        return v, h_out

##################################################
# ========== Server Actor/Critic Net ==========
##################################################
class server_GRUActorNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=32, use_gru=False):
        super().__init__()
        self.use_gru = use_gru
        self.mlp = nn.Linear(obs_dim, hidden_dim)
        if self.use_gru:
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, action_dim)
    def forward(self, obs, h_in=None):
        x = torch.relu(self.mlp(obs))
        if self.use_gru:
            x = x.unsqueeze(1)
            if h_in is None:
                h_in = torch.zeros(1, obs.size(0), x.size(-1), device=obs.device)
            x, h_out = self.gru(x, h_in)
            x = x.squeeze(1)
        else:
            h_out = h_in
        logits = self.fc_out(x)
        return logits, h_out

class server_GRUCriticNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim=32, use_gru=False):
        super().__init__()
        self.use_gru = use_gru
        self.mlp = nn.Linear(obs_dim, hidden_dim)
        if self.use_gru:
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)
    def forward(self, obs, h_in=None):
        x = torch.relu(self.mlp(obs))
        if self.use_gru:
            x = x.unsqueeze(1)
            if h_in is None:
                h_in = torch.zeros(1, obs.size(0), x.size(-1), device=obs.device)
            x, h_out = self.gru(x, h_in)
            x = x.squeeze(1)
        else:
            h_out = h_in
        v = self.fc_out(x)
        return v, h_out

class ServerActorNetNoGRUForELLA(nn.Module):
    """
    Example: a server actor that has a separate 'policy_mlp' (the parameters
    we want to treat with ELLA) plus an optional GRU that we do NOT update
    with ELLA.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=32, use_gru=False):
        super().__init__()
        self.use_gru = use_gru

        # LayerNorm for input stability
        self.obs_norm = nn.LayerNorm(obs_dim)

        # "policy" parameters: factorized by ELLA
        self.input_mlp = nn.Linear(obs_dim, hidden_dim)  # <-- ELLA parameters
        # feature_norm removed to keep the structure simple

        self.policy_mlp = nn.Linear(hidden_dim, hidden_dim)

        # optional GRU (excluded from ELLA)
        if self.use_gru:
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        else:
            self.gru = None

        # final output head
        self.fc_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs, h_in=None):
        """
        obs: shape (batch, obs_dim)
        h_in: optional hidden state for the GRU
        """
        if torch.isnan(obs).any():
            raise ValueError("NaN detected in input observations")

        # LayerNorm to stabilize the input features
        obs_norm = self.obs_norm(obs)
        feature = torch.tanh(self.input_mlp(obs_norm))
        if torch.isnan(feature).any():
            raise ValueError("NaN detected after input_mlp")

        x = torch.tanh(self.policy_mlp(feature))
        if torch.isnan(x).any():
            raise ValueError("NaN detected after policy_mlp")

        if self.gru is not None:
            x = x.unsqueeze(1)
            if h_in is None:
                h_in = torch.zeros(1, obs.size(0), x.size(-1), device=obs.device)
            x, h_out = self.gru(x, h_in)
            x = x.squeeze(1)
        else:
            h_out = h_in

        logits = self.fc_out(x)
        if torch.isnan(logits).any():
            raise ValueError("NaN detected after fc_out")

        return logits, h_out, feature

    def extract_policy_parameters(self):
        """
        Return the raw weight and bias from the MLP that ELLA will manipulate.
        We'll flatten them as a single vector 'theta'.
        """
        w = self.policy_mlp.weight.detach().view(-1, 1)  # shape (hidden_dim * obs_dim, 1)
        b = self.policy_mlp.bias.detach().view(-1, 1)    # shape (hidden_dim, 1)
        return torch.cat([w, b], dim=0)              # shape (W+B, 1)

    def load_policy_parameters(self, theta):
        """
        Inverse of extract_policy_parameters().  We reshape and reassign
        the MLP weight/bias from the combined vector.
        """
        obs_dim = self.policy_mlp.weight.shape[1]
        hidden_dim = self.policy_mlp.weight.shape[0]

        num_w = obs_dim * hidden_dim
        w_new = theta[:num_w].view(hidden_dim, obs_dim)
        b_new = theta[num_w:].view(hidden_dim)

        with torch.no_grad():
            self.policy_mlp.weight.copy_(w_new)
            self.policy_mlp.bias.copy_(b_new)


class ServerActorNetNoGRUForELLA_Abstract_feature(nn.Module):
    """
    Modified: Treats the abstract feature vector output by policy_mlp as the ELLA parameters.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=32, use_gru=False):
        super().__init__()
        self.use_gru = use_gru

        # LayerNorm for input stability
        self.obs_norm = nn.LayerNorm(obs_dim)

        # input processing (kept unchanged)
        self.input_mlp = nn.Linear(obs_dim, hidden_dim)

        # abstract feature extraction (target for ELLA)
        self.policy_mlp = nn.Linear(hidden_dim, hidden_dim)

        # optional GRU (excluded from ELLA)
        if self.use_gru:
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        else:
            self.gru = None

        # final output head
        self.fc_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs, h_in=None):
        if torch.isnan(obs).any():
            raise ValueError("NaN detected in input observations")

        obs_norm = self.obs_norm(obs)
        feature = torch.tanh(self.input_mlp(obs_norm))
        if torch.isnan(feature).any():
            raise ValueError("NaN detected after input_mlp")

        # Abstract feature vector (for ELLA)
        abstract_feature = torch.tanh(self.policy_mlp(feature))
        if torch.isnan(abstract_feature).any():
            raise ValueError("NaN detected after policy_mlp")

        if self.gru is not None:
            x = abstract_feature.unsqueeze(1)
            if h_in is None:
                h_in = torch.zeros(1, obs.size(0), x.size(-1), device=obs.device)
            x, h_out = self.gru(x, h_in)
            x = x.squeeze(1)
        else:
            x = abstract_feature
            h_out = h_in

        logits = self.fc_out(x)
        if torch.isnan(logits).any():
            raise ValueError("NaN detected after fc_out")

        return logits, h_out, abstract_feature

    def extract_policy_parameters(self):
        # Dummy input used only to infer the feature dimension
        dummy_obs = torch.zeros(1, self.obs_norm.normalized_shape[0]).to(next(self.parameters()).device)
        _, _, abstract_feature = self(dummy_obs)
        return abstract_feature.view(-1, 1)

    def load_policy_parameters(self, abstract_feature_vector, batch_size):
        """
        Loads the abstract feature vector back (reshapes if needed). Typically,
        this vector might be directly used or projected into policy_mlp.
        """
        hidden_dim = self.policy_mlp.out_features
        abstract_feature = abstract_feature_vector.view(batch_size, hidden_dim)
        return abstract_feature
