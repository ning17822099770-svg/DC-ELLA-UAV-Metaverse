"""PPO local learner of each UAV server (base learner of DC-ELLA, Sec. V-C).

ServerMAPPOAgent holds one actor-critic per UAV. The action vector holds the
user-server association (first I=15 dims) followed by subject and object cache
decisions (N=K=50); a per-block Top-10 rule enforces the cache capacity N_F.

Interface to the lifelong learner: update_single() runs the PPO update and
returns (actor_loss, critic_loss, total_loss, H_m), where H_m is a curvature
estimate of the policy parameters (empirical Fisher, Gauss-Newton or diagonal
approximation, see _compute_actor_*). DC-ELLA uses theta_m and H_m in
Eq. 23-26.

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

from network_define import ServerActorNetNoGRUForELLA, server_GRUCriticNet, ServerActorNetNoGRUForELLA_Abstract_feature


def print_tensor_properties(name, tensor):
    # If the input is a PyTorch tensor, convert it to a NumPy array.
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    # Compute basic statistics regardless of dimensionality.
    mean_val = np.mean(tensor)
    max_val = np.max(tensor)
    min_val = np.min(tensor)

    # Initialize eigenvalue and condition number variables.
    eigen_min = None
    eigen_max = None
    cond_num = None

    # Only compute eigenvalues and condition number if tensor is 2D and square.
    if tensor.ndim == 2 and tensor.shape[0] == tensor.shape[1]:
        tol = 1e-6  # tolerance for numerical precision
        # Check if the matrix is (close to) symmetric.
        if np.allclose(tensor, tensor.T, atol=tol):
            eigenvalues = np.linalg.eigvalsh(tensor)
        else:
            eigenvalues = np.linalg.eigvals(tensor)
            # If eigenvalues are complex but with negligible imaginary parts, take the real parts.
            if np.iscomplexobj(eigenvalues) and np.max(np.abs(eigenvalues.imag)) < tol:
                eigenvalues = eigenvalues.real
            elif np.iscomplexobj(eigenvalues):
                print(f"Warning: {name} has significant imaginary parts in its eigenvalues.")

        eigen_min = eigenvalues.min()
        eigen_max = eigenvalues.max()
        cond_num = np.linalg.cond(tensor)

    # Print the computed properties.
    # For eigenvalues and condition number, print "N/A" if not computed.
    eigen_min_str = f"{eigen_min:.4f}" if eigen_min is not None else "N/A"
    eigen_max_str = f"{eigen_max:.4f}" if eigen_max is not None else "N/A"
    cond_num_str = f"{cond_num:.4e}" if cond_num is not None else "N/A"

    print(f"{name}: mean = {mean_val:.4f}, max = {max_val:.4f}, min = {min_val:.4f}, "
          f"eigen_min = {eigen_min_str}, eigen_max = {eigen_max_str}, "
          f"condition_number = {cond_num_str}")


class ServerMAPPOAgent:
    def __init__(self, num_servers=25, obs_dim=40, action_dim=100,
                 hidden_dim=32, lr=1e-4, gamma=0.99, lam=0.95,
                 alpha=0.5, beta=0.5, device=torch.device("cpu"),
                 clip_ratio=0.2, entropy_coef=0.01, max_grad_norm=0.5):
        self.device = device
        self.num_servers = num_servers
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.alpha = alpha
        self.beta = beta
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.actors = []
        self.critics = []
        for _ in range(num_servers):
            actor = ServerActorNetNoGRUForELLA(obs_dim, action_dim, hidden_dim).to(self.device)
            # actor = ServerActorNetNoGRUForELLA_Abstract_feature(obs_dim, action_dim, hidden_dim).to(self.device)


            critic = server_GRUCriticNet(obs_dim, hidden_dim).to(self.device)
            self.actors.append(actor)
            self.critics.append(critic)
        self.actor_h = [None] * num_servers
        self.critic_h = [None] * num_servers
        params = []
        for i in range(num_servers):
            params += list(self.actors[i].parameters())
            params += list(self.critics[i].parameters())
        self.optimizer = optim.Adam(params, lr=lr)
    def reset_hidden_states(self):
        self.actor_h = [None] * self.num_servers
        self.critic_h = [None] * self.num_servers
    @torch.no_grad()
    def get_actions_all_servers(self, obs_list):
        k = 10
        acts = []
        logps = []
        vals = []
        for i in range(self.num_servers):
            obs_i = obs_list[i]
            obs_t = torch.tensor(obs_i, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, h_new, _ = self.actors[i](obs_t, self.actor_h[i])
            self.actor_h[i] = h_new
            probs = torch.sigmoid(logits)
            dist_bern = Bernoulli(probs)
            raw_a_samp = dist_bern.sample()
            sub_samp = raw_a_samp[:, :50]
            sub_probs = probs[:, :50]
            sub_count = int(sub_samp.sum().item())
            if sub_count > k:
                ones_idx = (sub_samp[0] == 1).nonzero(as_tuple=True)[0]
                selected_probs = sub_probs[0, ones_idx]
                topk_values, topk_indices = torch.topk(selected_probs, k)
                keep_idx = ones_idx[topk_indices]
                sub_samp[0, :] = 0.0
                sub_samp[0, keep_idx] = 1.0
            obj_samp = raw_a_samp[:, 50:100]
            obj_probs = probs[:, 50:100]
            obj_count = int(obj_samp.sum().item())
            if obj_count > k:
                ones_idx = (obj_samp[0] == 1).nonzero(as_tuple=True)[0]
                selected_probs = obj_probs[0, ones_idx]
                topk_values, topk_indices = torch.topk(selected_probs, k)
                keep_idx = ones_idx[topk_indices]
                obj_samp[0, :] = 0.0
                obj_samp[0, keep_idx] = 1.0
            final_a_samp = torch.cat([sub_samp, obj_samp], dim=1)
            logp_final = dist_bern.log_prob(final_a_samp).sum(dim=1)
            v_t, h_c_new = self.critics[i](obs_t, self.critic_h[i])
            self.critic_h[i] = h_c_new
            act_i = final_a_samp[0].detach().cpu().numpy().astype(int)
            logp_i = logp_final.item()
            val_i = v_t.item()
            acts.append(act_i)
            logps.append(logp_i)
            vals.append(val_i)
        return acts, logps, vals
    def decode_action(self, action_100):
        subject_ids = []
        object_ids = []
        for s in range(50):
            if action_100[s] == 1:
                subject_ids.append(s + 1)
        for o in range(50):
            if action_100[50 + o] == 1:
                object_ids.append(o + 51)
        return subject_ids, object_ids
    def compute_logp_value_entropy_single_server(self, obs_batch, act_batch, server_i,
                                                 rnn_actor_h=None, rnn_critic_h=None):
        T = obs_batch.size(0)
        logp_list = []
        ent_list = []
        v_list = []
        h_a = rnn_actor_h
        h_c = rnn_critic_h
        for t in range(T):
            obs_t = obs_batch[t].unsqueeze(0)
            act_t = act_batch[t].float().unsqueeze(0)
            logits, h_a, _ = self.actors[server_i](obs_t, h_a)
            probs = torch.sigmoid(logits)
            dist_bern = Bernoulli(probs)
            lp_t = dist_bern.log_prob(act_t).sum(dim=1)
            ent_t = dist_bern.entropy().sum(dim=1)
            v_out, h_c = self.critics[server_i](obs_t, h_c)
            logp_list.append(lp_t[0])
            ent_list.append(ent_t[0])
            v_list.append(v_out[0, 0])
        logp_all = torch.stack(logp_list, dim=0)
        ent_all = torch.stack(ent_list, dim=0)
        v_all = torch.stack(v_list, dim=0)
        return logp_all, ent_all, v_all, h_a, h_c

    def update_single(self, server_i, single_buf, train_iters=20, batch_size=128):
        obs_np = single_buf["obs"]
        T = obs_np.shape[0]
        if T == 0:
            return 0.0, 0.0, 0.0
        obs_dim = obs_np.shape[2]
        obs_flat = obs_np.reshape(T, obs_dim)
        obs_t = torch.tensor(obs_flat, dtype=torch.float32, device=self.device)
        act_t = torch.tensor(single_buf["actions"], dtype=torch.float32, device=self.device)
        logp_old_t = torch.tensor(single_buf["logp"], dtype=torch.float32, device=self.device).view(T)
        adv_t = torch.tensor(single_buf["adv"], dtype=torch.float32, device=self.device).view(T)
        ret_t = torch.tensor(single_buf["returns"], dtype=torch.float32, device=self.device).view(T)
        partial_rews_t = torch.tensor(single_buf["partial_rews"], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, _, v_seq, _, _ = self.compute_logp_value_entropy_single_server(obs_t, act_t, server_i, None, None)
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
        for epoch_iter in range(train_iters):
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
                partial_b = partial_rews_t[mb]
                logp_new_b, ent_new_b, v_new_b, _, _ = self.compute_logp_value_entropy_single_server(obs_b, act_b, server_i, None, None)
                ratio = torch.exp(logp_new_b - logp_old_b)
                unclipped = ratio * adv_b
                clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_b
                loss_actor_ppo = - torch.mean(torch.min(unclipped, clipped))
                logits2, _, _ = self.actors[server_i](obs_b, None)
                probs2 = torch.sigmoid(logits2)
                dist_bern = Bernoulli(probs2)
                logp_mdim = dist_bern.log_prob(act_b)
                alpha_partial = 0.1
                partial_loss = - (partial_b * logp_mdim).sum(dim=1).mean()
                loss_actor = loss_actor_ppo + alpha_partial * partial_loss
                clip_v = 0.1 #0.01
                v_clipped = v_old_b + torch.clamp(v_new_b - v_old_b, -clip_v, clip_v)
                c_unclip = (ret_b - v_new_b)**2
                c_clip = (ret_b - v_clipped)**2

                if epoch_iter % 10 ==0:
                    # print('ret_b', ret_b, 'c_unclip = ', c_unclip, 'c_clip = ', c_clip)
                    print_tensor_properties('ret_b', ret_b)
                    print_tensor_properties('c_unclip', c_unclip)
                    print_tensor_properties('c_clip', c_clip)

                loss_critic = torch.mean(torch.max(c_unclip, c_clip))
                ent = torch.mean(ent_new_b)
                loss_entropy = - self.entropy_coef * ent
                loss = loss_actor + 0.5 * loss_critic + loss_entropy
                self.optimizer.zero_grad()
                loss.backward()

                # region Log Grad Norms
                total_norm = 0.0
                for p in self.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print("Grad norm pre-clipping:", total_norm)
                # endregion

                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total_actor_loss += loss_actor.item()
                total_critic_loss += loss_critic.item()
                total_loss += loss.item()
        avg_actor = total_actor_loss / total_updates
        avg_critic = total_critic_loss / total_updates
        avg_loss = total_loss / total_updates
        with torch.no_grad():
            ret_mean = ret_b.mean().item()
            v_mean = v_new_b.mean().item()
        print(f"[update_single] Server={server_i + 1}, CriticLoss={avg_critic:.3f}, totalLoss={avg_loss:.3f}")
        print(f"   => ret_mean={ret_mean:.3f}, v_mean={v_mean:.3f}, actor_loss={avg_actor:.3f}")

        # ============ Hessian-Related: approximate Hessian after final PPO ==============
        local_hess = self._compute_actor_diagonal_hessian_approx(server_i, obs_t, act_t)

        return avg_actor, avg_critic, avg_loss, local_hess

    def parameters(self):
        for i in range(self.num_servers):
            yield from self.actors[i].parameters()
            yield from self.critics[i].parameters()

    def _extract_actor_grad(self, server_i):
        """
        Grabs the gradient from the policy_mlp (weight + bias),
        concatenates into a single 1D vector.
        """
        w_grad = self.actors[server_i].policy_mlp.weight.grad.view(-1)
        b_grad = self.actors[server_i].policy_mlp.bias.grad.view(-1)
        return torch.cat([w_grad, b_grad], dim=0)

    def _compute_actor_empirical_fisher_hessian(self, server_i, obs_t, act_t):
        """
        Empirical Fisher
        Approximate Hessian for the negative log-likelihood of the final,
        updated actor.  We do a per-sample gradient, then sum outer products.

        - obs_t: shape (T, obs_dim)
        - act_t: shape (T, action_dim) (Bernoulli 0/1 in each dimension)
        Returns a (d x d) numpy array, where d = # actor policy parameters.
        """
        actor = self.actors[server_i]
        d = actor.extract_policy_parameters().numel()
        H = torch.zeros(d, d, device=self.device)
        T = obs_t.size(0)
        # We'll do a loop over each sample; for large T you can also
        # random-subsample or do mini-batches to avoid big overhead.
        for i in range(T):
            # single sample
            o_i = obs_t[i].unsqueeze(0)  # shape (1, obs_dim)
            a_i = act_t[i].unsqueeze(0)  # shape (1, action_dim)
            # forward
            # In _compute_actor_hessian, zero the gradients before the forward pass
            actor.zero_grad()
            logits, _, _ = actor(o_i)
            probs = torch.sigmoid(logits)
            dist_bern = Bernoulli(probs)
            nll_i = - dist_bern.log_prob(a_i).sum()  # negative log-likelihood
            # compute grad wrt policy parameters only
            nll_i.backward(retain_graph=False)
            # Extract gradient as a flat vector d_θ
            g = self._extract_actor_grad(server_i)  # shape (d,)
            # outer product
            H += torch.ger(g, g)  # g g^T
        # (1) Normalize by T for an empirical average
        H /= float(T)
        # (2) Optionally, add a small ridge to ensure positivity
        H += 1e-6 * torch.eye(d, device=self.device)
        return H.detach().cpu().numpy()

    def _compute_actor_gauss_newton_hessian(self, server_i, obs_t, act_t):
        """
        Compute the Gauss-Newton approximation for the Hessian of the negative
        log-likelihood with respect to the policy parameters (i.e. the parameters
        of policy_mlp) for a given actor.

        For a single sample with logits ℓ, note that for Bernoulli cross entropy
        loss, the second derivative with respect to the logits is:

             d²L/dℓ² = σ(ℓ)(1-σ(ℓ))

        If J is the Jacobian of the logits with respect to the policy parameters,
        then the Gauss–Newton approximation is:

             H_GN = Jᵀ diag(σ(ℓ)(1-σ(ℓ))) J.

        We accumulate this per sample and average over the T samples.

        Args:
            server_i: index to select the actor.
            obs_t (Tensor): shape (T, obs_dim)
            act_t (Tensor): shape (T, action_dim)

        Returns:
            A (d x d) numpy array, where d = number of policy_mlp parameters.
        """
        actor = self.actors[server_i]
        device = self.device
        # d: total number of parameters in policy_mlp (weight and bias)
        d = actor.extract_policy_parameters().numel()
        H = torch.zeros(d, d, device=device)
        T = obs_t.size(0)

        # We will compute derivatives only with respect to policy_mlp parameters.
        # Define the list of parameters we care about:
        params = [actor.policy_mlp.weight, actor.policy_mlp.bias]

        for i in range(T):
            # Get the i-th sample.
            o_i = obs_t[i].unsqueeze(0)  # shape: (1, obs_dim)
            # (a_i is not needed in the GN approximation because for cross entropy
            # the Hessian with respect to the logits does not depend on the label.)
            # a_i = act_t[i].unsqueeze(0)  # shape: (1, action_dim)

            # Clear gradients from previous iterations.
            actor.zero_grad()

            # Forward pass: obtain logits.
            logits, _, _ = actor(o_i)  # logits shape: (1, action_dim)

            # Compute probabilities using the sigmoid.
            probs = torch.sigmoid(logits)  # shape: (1, action_dim)
            # The diagonal of the second derivative (d²L/dℓ²) is:
            D = (probs * (1 - probs)).squeeze(0)  # shape: (action_dim,)

            # We now compute the Jacobian J of the logits with respect to the policy_mlp parameters.
            # J will have shape (action_dim, d) where each row j is:
            #   J[j, :] = ∇_θ logit_j
            J_rows = []
            num_outputs = logits.size(1)
            for j in range(num_outputs):
                # Compute gradient of logits[0, j] with respect to the policy_mlp parameters.
                # We use retain_graph=True since we need to call autograd.grad repeatedly on the same graph.
                grad_j = torch.autograd.grad(logits[0, j], params, retain_graph=True, allow_unused=True)
                # In case some parameter does not contribute to this logit, replace None with zeros.
                grad_j = [
                    g if g is not None else torch.zeros_like(p)
                    for g, p in zip(grad_j, params)
                ]
                # Flatten each gradient and concatenate.
                grad_j_flat = torch.cat([g.reshape(-1) for g in grad_j])
                # Append as a new row in the Jacobian.
                J_rows.append(grad_j_flat.unsqueeze(0))  # shape: (1, d)
            # Stack the rows to form the full Jacobian J (shape: (action_dim, d)).
            J = torch.cat(J_rows, dim=0)

            # Compute the Gauss–Newton contribution for this sample:
            #   H_sample = Jᵀ * diag(D) * J
            weighted_J = J * D.unsqueeze(1)  # Each row j is scaled by D[j]
            H_sample = J.t().mm(weighted_J)  # shape: (d, d)
            H += H_sample

        # Average the Hessian over all T samples.
        H /= float(T)
        # Optionally, add a small ridge for numerical stability.
        H += 1e-6 * torch.eye(d, device=device)
        return H.detach().cpu().numpy()

    def _compute_actor_diagonal_hessian_approx(self, server_i, obs_t, act_t):
        """
        Compute a diagonal Gauss-Newton approximation of the Hessian of the
        negative log-likelihood for a Bernoulli policy.

        Similar to `_compute_actor_gauss_newton_hessian`, but we only keep the
        diagonal entries of Jᵀ diag(σ(ℓ)(1−σ(ℓ))) J, effectively creating a
        diagonal Hessian approximation.

        Returns:
            A (d x d) numpy array where d is the number of policy_mlp parameters.
            Only the diagonal entries are non-zero.
        """
        actor = self.actors[server_i]
        device = self.device

        # d: total number of parameters in the actor's policy_mlp
        d = actor.extract_policy_parameters().numel()
        # We'll accumulate diagonal entries in a 1D buffer, then place
        # them in a diagonal matrix at the end.
        diag_H = torch.zeros(d, device=device)
        T = obs_t.size(0)

        # We will compute derivatives only with respect to policy_mlp parameters:
        params = [actor.policy_mlp.weight, actor.policy_mlp.bias]

        for i in range(T):
            # Single sample
            o_i = obs_t[i].unsqueeze(0)  # shape: (1, obs_dim)
            actor.zero_grad()

            logits, _, _ = actor(o_i)      # shape: (1, action_dim)
            probs = torch.sigmoid(logits)  # shape: (1, action_dim)
            # For Bernoulli cross-entropy, diagonal of Hessian w.r.t. logits is σ(ℓ)(1 - σ(ℓ)).
            D = (probs * (1 - probs)).squeeze(0)  # shape: (action_dim,)

            # Compute the Jacobian J of 'logits' wrt the policy parameters.
            # We'll need one gradient per action_dim output:
            J_rows = []
            num_outputs = logits.size(1)  # action_dim
            for j in range(num_outputs):
                grad_j = torch.autograd.grad(logits[0, j], params,
                                             retain_graph=True, allow_unused=True)
                # Replace None with zeros (in case some param doesn't affect this output)
                grad_j = [
                    g if g is not None else torch.zeros_like(p)
                    for g, p in zip(grad_j, params)
                ]
                # Flatten each gradient and concatenate into one row
                grad_j_flat = torch.cat([g.reshape(-1) for g in grad_j])
                J_rows.append(grad_j_flat.unsqueeze(0))  # (1, d)
            # Stack the rows to form J: (action_dim, d)
            J = torch.cat(J_rows, dim=0)

            # For the diagonal approximation:
            # diag(H_sample) = Σ_j [ D_j * J[j,:]^2 ]
            # So we multiply each row of J^2 by D_j and sum across j.
            J_sq = J ** 2  # shape: (action_dim, d)
            J_sq_weighted = J_sq * D.unsqueeze(1)  # multiply each row j by D_j
            # Sum across j to get the diagonal contribution for each parameter.
            diag_H += J_sq_weighted.sum(dim=0)

        # Average over T samples
        diag_H /= float(T)
        # Add a ridge for numerical stability
        diag_H += 0.1 #0.002

        # Create a full (d x d) matrix with diag_H on its diagonal
        H_diag_matrix = torch.diag(diag_H).detach().cpu().numpy()
        return H_diag_matrix

    # def _compute_actor_diagonal_hessian_approx(self, server_i, obs_t, act_t):
    #     """
    #     Compute a diagonal Gauss-Newton approximation of the Hessian of the
    #     negative log-likelihood for a Bernoulli policy, using the abstract feature
    #     vector as parameters.
    #
    #     Returns:
    #         A (d x d) numpy array where d is the number of elements in the abstract feature vector.
    #         Only the diagonal entries are non-zero.
    #     """
    #     actor = self.actors[server_i]
    #     device = self.device
    #
    #     T = obs_t.size(0)
    #
    #     diag_H = None
    #
    #     for i in range(T):
    #         # Single sample
    #         o_i = obs_t[i].unsqueeze(0).to(device)  # shape: (1, obs_dim)
    #         actor.zero_grad()
    #
    #         _, _, abstract_feature = actor(o_i)
    #
    #         # Enable gradients explicitly for abstract_feature
    #         abstract_feature = abstract_feature.detach().requires_grad_(True)
    #
    #         # Recompute logits explicitly dependent on abstract_feature
    #         if actor.gru is not None:
    #             x = abstract_feature.unsqueeze(1)
    #             h_in = torch.zeros(1, o_i.size(0), x.size(-1), device=device)
    #             x, _ = actor.gru(x, h_in)
    #             x = x.squeeze(1)
    #         else:
    #             x = abstract_feature
    #
    #         logits = actor.fc_out(x)
    #         probs = torch.sigmoid(logits).squeeze(0)  # shape: (action_dim,)
    #         D = probs * (1 - probs)
    #
    #         params = [abstract_feature]
    #         num_outputs = logits.size(1)
    #         J_rows = []
    #
    #         for j in range(num_outputs):
    #             grad_j = torch.autograd.grad(logits[0, j], params,
    #                                          retain_graph=True, allow_unused=True)
    #
    #             grad_j = [
    #                 g if g is not None else torch.zeros_like(p)
    #                 for g, p in zip(grad_j, params)
    #             ]
    #
    #             grad_j_flat = torch.cat([g.reshape(-1) for g in grad_j])
    #             J_rows.append(grad_j_flat.unsqueeze(0))
    #
    #         J = torch.cat(J_rows, dim=0)
    #         J_sq_weighted = (J ** 2) * D.unsqueeze(1)
    #
    #         if diag_H is None:
    #             diag_H = J_sq_weighted.sum(dim=0)
    #         else:
    #             diag_H += J_sq_weighted.sum(dim=0)
    #
    #     diag_H /= float(T)
    #     diag_H += 0.1  # Regularization for numerical stability
    #
    #     H_diag_matrix = torch.diag(diag_H).detach().cpu().numpy()
    #     return H_diag_matrix
