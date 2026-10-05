"""PG-ELLA baseline lifelong learner (no environment-feature dictionary D).

Same interface as Server_LL_Learner_with_dictionary.py, but s_m is solved from
the policy part only (solve_for_s), i.e. the PG-ELLA comparison in Sec. VI-B
(Fig. 7). Used by baseline_pg_ella.py.

Paper: "Deep Lifelong Learning for Adaptive Semantic-Aware Content Reuse in
UAV-Assisted Metaverse", IEEE Transactions on Mobile Computing.
Equation / section numbers refer to the paper.
"""

# Server_LL_Learner.py

import numpy as np
import random
from numpy.linalg import norm
from scipy.linalg import sqrtm
from sklearn.linear_model import Lasso  # example
import cvxpy as cp
from cvxpy import Variable, Minimize, sum_squares, norm as cvx_norm, Problem
# import spams
import joblib


def vec(M):
    """Vectorize a 2D numpy array (column-major order)."""
    return M.reshape((-1, 1), order='F')

def mat(v, nrows, ncols):
    """Inverse of vec: shape (nrows, ncols) using column-major order."""
    return v.reshape((nrows, ncols), order='F')

class ServerLifelongLearner:
    """
    Example ELLA structure to maintain a global dictionary L (size d x K)
    and per-server s_i (size K x 1).
    - dict_L: {domain_id -> L matrix}, or just a single L if you assume all servers share same domain
    - dict_s: {server_id -> s vector}
    - dict_hess: {server_id -> approximate Hessian for that server’s local policy}
    - dict_theta: {server_id -> last known local policy parameters (for reference)}
    """

    def __init__(self, dim_theta, dim_feature, dim_latent=10, mu_one=0.001, mu_two=0.001, lr=0.001, integration_rate=1.0):
        """
        dim_theta: dimension of the "policy parameter" vector for each server
        dim_latent: number of latent components K
        mu_one, mu_two: ELLA’s LASSO and Frobenius regularization strengths
        lr: step size for updating L
        integration_rate: the beta value for gradual integration (0 < beta <= 1)
        """
        # -- Load the offline-fitted scaler and PCA (see fit_feature_scaler_pca.py) -- #
        # Raises FileNotFoundError if the files are missing (run from the experiment folder)
        self.scaler = joblib.load("feature_scaler.pkl")
        self.pca    = joblib.load("feature_pca.pkl")

        self.task_feature_samples = {}  # {task_id: [feat1, feat2, ...]}, used to update cluster centers
        self.max_samples = 20  # max samples kept per task (tunable)
        self.dim_theta = dim_theta
        self.dim_feature = dim_feature
        self.dim_latent = dim_latent
        self.mu_one = mu_one
        self.mu_two = mu_two
        self.learning_rate = lr
        self.integration_rate = integration_rate

        # A single global dictionary L, shape (dim_theta, dim_latent)
        # If you want separate L per domain, store them in a dict.
        self.L = np.random.randn(dim_theta, dim_latent).astype(np.float32) * 2

        # Per-server expansions:
        self.dict_s = {}  # {server_id -> s vector of shape (dim_latent, 1)}
        self.dict_hess = {}  # {server_id -> Hessian approx: shape (dim_theta, dim_theta)}
        self.dict_theta = {}  # {server_id -> last local policy param (theta_i) shape (dim_theta, 1)}

        # # Keep track of which servers we have observed
        # self.server_observed = set()

        self.task_observed = set()

        # PG-ELLA parameters
        self.T_count = 0  # how many distinct tasks/servers we have seen

        # A and b for the closed-form solution of L
        # Shape of A is (dim_latent*dim_theta, dim_latent*dim_theta)
        # Shape of b is (dim_latent*dim_theta, 1)
        self.A = np.zeros((self.dim_latent * self.dim_theta,
                           self.dim_latent * self.dim_theta), dtype=np.float32)
        self.b = np.zeros((self.dim_latent * self.dim_theta, 1), dtype=np.float32)

        # Optionally store alpha(t), Gamma(t), s(t) for each server
        self.dict_alpha = {}
        self.dict_gamma = {}
        # We already store s in self.dict_s; just keep it up-to-date

        self.task_feature_dict = {}  # mean feature vector of each task
        self.task_server_mapping = {}  # list of server IDs belonging to each task
        self.D = np.random.rand(self.dim_feature, self.dim_latent)


    # def is_observed_server(self, sid):
    #     return (sid in self.server_observed)
    #
    # def init_server_if_new(self, sid):
    #     if not self.is_observed_server(sid):
    #         self.server_observed.add(sid)
    #         # Initialize s for this server
    #         self.dict_s[sid] = np.zeros((self.dim_latent, 1))
    #         # A default Hessian identity or something
    #         self.dict_hess[sid] = np.eye(self.dim_theta)
    #         self.dict_theta[sid] = np.zeros((self.dim_theta, 1))

    def is_observed_task(self, task_id):
        return (task_id in self.task_observed)

    def init_task_if_new(self, task_id):
        if not self.is_observed_task(task_id):
            self.task_observed.add(task_id)
            self.T_count += 1

            # Initialize s, alpha, Gamma, theta for the task
            self.dict_s[task_id] = np.zeros((self.dim_latent, 1))
            self.dict_alpha[task_id] = np.zeros((self.dim_theta, 1))
            self.dict_gamma[task_id] = np.eye(self.dim_theta)
            self.dict_theta[task_id] = np.zeros((self.dim_theta, 1))

    def randomize_zero_columns(self, L):
        """
        Randomizes columns in matrix L which are all zero or contain inf/NaN values.
        """
        for col in range(L.shape[1]):
            if np.all(L[:, col] == 0) or \
                    np.any(np.isinf(L[:, col])) or \
                    np.any(np.isnan(L[:, col])):
                L[:, col] = np.random.rand(L.shape[0])
        return L

    def normalize_columns(self, L):
        """
        Normalize each column of L to have unit Euclidean norm.
        Returns the modified L.
        """
        for col_idx in range(L.shape[1]):
            col_norm = np.linalg.norm(L[:, col_idx])
            # If the column is non-zero, normalize it.
            if col_norm > 1e-12:
                L[:, col_idx] /= col_norm
        return L

    def remove_old_contribution(self, sid):
        """
        If we have old alpha, Gamma, s for server 'sid', remove them from A, b.
        This follows lines 10-11 in Algorithm 1 of PG-ELLA.
        """
        if sid not in self.dict_alpha or sid not in self.dict_gamma:
            return  # No old data to remove

        alpha_old = self.dict_alpha[sid]
        Gamma_old = self.dict_gamma[sid]
        s_old = self.dict_s[sid]

        # A -= (s_old s_old^T) \otimes Gamma_old
        s_outer = s_old @ s_old.T  # shape (k, k)
        # We do Kronecker: shape (k*d, k*d)
        A_contrib = np.kron(s_outer, Gamma_old)
        self.A -= A_contrib

        # b -= vec( s_old^T \otimes (alpha_old^T Gamma_old) )
        # shape of alpha_old^T Gamma_old is (1, d), so we do s_old^T \otimes that.
        alphaGamma = alpha_old.T @ Gamma_old  # shape (1, d)
        b_contrib = vec(np.kron(s_old.T, alphaGamma))
        self.b -= b_contrib

    def add_new_contribution(self, alpha_new, Gamma_new, s_new):
        """
        Add new data from a single server t to A and b (lines 16-17 in Algorithm 1).
        """
        s_outer = s_new @ s_new.T
        A_contrib = np.kron(s_outer, Gamma_new)
        self.A += A_contrib

        alphaGamma = alpha_new.T @ Gamma_new  # shape (1, d)
        b_contrib = vec(np.kron(s_new.T, alphaGamma))
        self.b += b_contrib

    def update_L_closed_form(self, if_rand_zero_col=False, if_norm_col=False, if_partial=False, alpha=0.3):
        """
        L <- mat( (1/T * A + lambda I)^(-1) (1/T * b ) ).
        We'll reinitialize zero columns after solving.
        """
        # shape is (dim_latent*dim_theta, dim_latent*dim_theta)
        dim_vec = self.dim_latent * self.dim_theta
        T = max(self.T_count, 1)  # avoid division by zero if T_count=0

        # Construct (1/T * A + lambda I)
        A_scaled = (1.0 / T) * self.A
        I_mat = np.eye(dim_vec, dtype=np.float32)
        # Add regularization \lambda I
        M = A_scaled + self.mu_two * I_mat

        # Right-hand side: (1/T) * b
        b_scaled = (1.0 / T) * self.b

        # Solve the system
        try:
            vec_L = np.linalg.solve(M, b_scaled)
        except np.linalg.LinAlgError:
            # fallback if M is singular or ill-conditioned
            vec_L, _, _, _ = np.linalg.lstsq(M, b_scaled, rcond=None)

        # Un-vectorize to get L of shape (dim_theta, dim_latent)
        # self.L = mat(vec_L, nrows=self.dim_theta, ncols=self.dim_latent)
        L_closed_form = mat(vec_L, nrows=self.dim_theta, ncols=self.dim_latent)
        # Option 2: partial integration (see next section), then column-normalize:
        if if_partial:
            self.L = (1.0 - alpha) * self.L + alpha * L_closed_form
        else:
            # Or if you still do full replacement:
            self.L = L_closed_form
        # (Optional) re-initialize zero or NaN columns in L
        if if_rand_zero_col:
            self.L = self.randomize_zero_columns(self.L)

        if if_norm_col:
            self.L = self.normalize_columns(self.L)

    def learn_multitask_server_actors(self, server_task_list):
        """
        server_task_list holds (task_id, alpha_cur, hess_cur) tuples;
        task_id is the RL task ID inferred from the feature, not the server id.
        """
        random.shuffle(server_task_list)

        for (task_id, alpha_cur_task, hess_cur_task) in server_task_list:
            # Initialize a new task
            self.init_task_if_new(task_id)

            # Remove the old contribution, if any (Eq. 28-29)
            self.remove_old_contribution(task_id)

            # Update alpha and Gamma
            alpha_new = alpha_cur_task
            Gamma_new = hess_cur_task

            # Solve for the new s
            s_old = self.dict_s[task_id]
            s_new = self.solve_for_s(alpha_new, Gamma_new, s_old, task_id)

            # Add the new contribution to A, b (Eq. 30-31)
            self.add_new_contribution(alpha_new, Gamma_new, s_new)

            # Update L
            self.update_L_closed_form()

            # Store the new values
            self.dict_alpha[task_id] = alpha_new
            self.dict_gamma[task_id] = Gamma_new
            self.dict_s[task_id] = s_new

            # Update the final parameters
            reconstructed_theta = self.L @ s_new
            self.dict_theta[task_id] = (
                    (1.0 - self.integration_rate) * self.dict_theta[task_id]
                    + self.integration_rate * reconstructed_theta
            )

        # Analyze L (optional)
        self.analyze_global_knowledge_base()

    def solve_for_s(self, alpha, Gamma, s_old, sid):
        from scipy.linalg import sqrtm
        import cvxpy as cp

        Gamma = (Gamma + Gamma.T) / 2
        eigvals = np.linalg.eigvalsh(Gamma)
        min_eigval = np.min(eigvals)
        if min_eigval <= 0:
            Gamma += np.eye(Gamma.shape[0]) * (abs(min_eigval) + 1e-6)

        d_sqrt = np.real(sqrtm(Gamma)).astype(np.float32)
        target = d_sqrt @ alpha
        dict_trans = d_sqrt @ self.L

        k = self.dim_latent
        s_cvx = cp.Variable(k)
        objective = cp.Minimize(
            0.5 * cp.sum_squares(target.flatten() - dict_trans @ s_cvx)
            + self.mu_one * cp.norm(s_cvx, 1)
        )
        prob = cp.Problem(objective)

        try:
            prob.solve(solver=cp.SCS, verbose=True, max_iters=20000, eps=1e-6)
        except cp.SolverError:
            # solver failed, fallback
            prob.solve(solver=cp.ECOS, verbose=True, max_iters=20000, abstol=1e-6, reltol=1e-6)

        if s_cvx.value is None:
            s_sol = s_old
        else:
            s_sol = s_cvx.value.reshape(-1, 1)

        return s_sol

    # def solve_for_s(self, alpha, Gamma, s_old, sid):
    #     """
    #     Solve for s in the L1-regularized regression problem:
    #
    #       min_s   || sqrt(Gamma)*alpha - sqrt(Gamma)*L*s ||^2 / 2   +   mu_one * ||s||_1
    #
    #     Using CVXPY or scikit-learn's Lasso, etc.
    #     """
    #     from numpy.linalg import norm
    #     from scipy.linalg import sqrtm
    #     import cvxpy as cp
    #
    #     d_sqrt = np.real(sqrtm(Gamma)).astype(np.float32)# shape (dim_theta, dim_theta)
    #     target = d_sqrt @ alpha  # shape (dim_theta, 1)
    #     dict_trans = d_sqrt @ self.L  # shape (dim_theta, dim_latent)
    #
    #     # We want s of shape (dim_latent, 1)
    #     k = self.dim_latent
    #     s_cvx = cp.Variable(k)
    #     # objective = 0.5 * ||target - dict_trans*s_cvx||^2 + mu_one * ||s_cvx||_1
    #     objective = cp.Minimize(
    #         0.5 * cp.sum_squares(target.flatten() - dict_trans @ s_cvx)
    #         + self.mu_one * cp.norm(s_cvx, 1)
    #     )
    #     prob = cp.Problem(objective)
    #     # prob.solve(
    #     #     solver=cp.ECOS,
    #     #     max_iters = 10_000,
    #     #     abstol = 1e-9,
    #     #     reltol = 1e-9,
    #     #     feastol = 1e-9,
    #     #     verbose = True
    #     # )
    #     # or whichever solver
    #     # prob.solve(
    #     #     solver=cp.SCS,
    #     #     max_iters=10_000,
    #     #     eps=1e-8,
    #     #     alpha=1.5,
    #     #     verbose=True
    #     # )
    #     # prob.solve(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9, max_iter=5_000, verbose=False)
    #     prob.solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6, max_iter=10000, verbose=True)
    #
    #     if s_cvx.value is None:
    #         # fallback if solver fails
    #         s_sol = s_old
    #     else:
    #         s_sol = s_cvx.value.reshape(-1, 1)
    #
    #     # -- Call the new helper function to check solution accuracy --
    #     self.check_lasso_solution_accuracy(target, dict_trans, s_sol, self.mu_one, sid)
    #
    #     return s_sol

    def check_lasso_solution_accuracy(self, target, dict_trans, s_sol, mu_one, sid):
        """
        Check and return metrics about how accurately s_sol solves the Lasso problem:

          min_s  0.5 * ||target - dict_trans @ s||^2_2 + mu_one * ||s||_1

        Returns a dict with:
          - 'abs_residual'    : the absolute L2 norm of the residual
          - 'rel_residual'    : the relative residual
          - 'lasso_objective' : the Lasso objective value

        Parameters:
            target       : shape (d,) or (d,1) vector
            dict_trans   : shape (d, k) matrix
            s_sol        : shape (k,) or (k,1) solution vector
            mu_one       : regularization coefficient
        """
        # Ensure shapes are compatible
        target_1d = target.flatten()
        s_sol_1d = s_sol.flatten()

        # Compute residual r = target - dict_trans @ s
        residual = target_1d - dict_trans @ s_sol_1d

        # Compute norms
        abs_residual = np.linalg.norm(residual)
        target_norm = np.linalg.norm(target_1d) + 1e-12  # avoid /0
        rel_residual = abs_residual / target_norm

        # Compute the Lasso objective: 0.5 * ||r||^2 + mu_one * ||s_sol||_1
        lasso_objective = 0.5 * (abs_residual ** 2) + mu_one * np.sum(np.abs(s_sol_1d))

        # Optionally, print them out
        print(f"s_{sid}'s Lasso solution accuracy check:")
        print(f"  Absolute residual: {abs_residual:.4e}")
        print(f"  Relative residual: {rel_residual:.4e}")
        print(f"  Lasso objective value: {lasso_objective:.4e}")

        return {
            'abs_residual': abs_residual,
            'rel_residual': rel_residual,
            'lasso_objective': lasso_objective
        }

    def get_reconstructed_theta(self, sid):
        """
        Return the factorized local parameter L*s_i for server sid.
        """
        return self.dict_theta[sid]

    def get_reconstructed_theta_(self, sid, current_theta):
        """
        Return the integrated local parameter for server sid.
        Instead of fully replacing the current policy parameter with L*s_i,
        we use gradual integration:

          theta_new = (1 - beta) * current_theta + beta * (L * s_i)

        where beta is the integration rate (self.integration_rate).

        Parameters:
            sid: the server identifier.
            current_theta: the current policy parameters of the server (as a numpy array of shape (dim_theta, 1))
        Returns:
            integrated_theta: the new parameter vector after gradual integration.
        """
        reconstructed_theta = self.dict_theta[sid]
        # Ensure current_theta is in the same shape as reconstructed_theta
        integrated_theta = (1 - self.integration_rate) * current_theta + self.integration_rate * reconstructed_theta
        return integrated_theta

    def analyze_global_knowledge_base(self):
        """
        Analyze and print various properties of the global knowledge base L.
        """
        print("\n=== Global Knowledge Base L Analysis ===")
        print(f"L shape: {self.L.shape}")
        print(f"Mean: {self.L.mean():.4e}")
        print(f"Max: {self.L.max():.4e}")
        print(f"Min: {self.L.min():.4e}")
        frob_norm = np.linalg.norm(self.L, 'fro')
        print(f"Frobenius norm: {frob_norm:.4e}")
        cond_num = np.linalg.cond(self.L)
        print(f"Condition number: {cond_num:.4e}")

        # Compute singular values
        singular_values = np.linalg.svd(self.L, compute_uv=False)
        print(f"Singular values: {singular_values}")

        # Compute sparsity: fraction of elements that are near-zero (within a threshold)
        near_zero_threshold = 1e-6
        sparsity = np.mean(np.abs(self.L) < near_zero_threshold)
        print(f"Sparsity (fraction of near-zero entries): {sparsity:.4e}")

        # For a non-square L, consider the Gram matrix L^T L (which is square and positive semi-definite)
        gram = self.L.T @ self.L
        eigvals = np.linalg.eigvalsh(gram)
        print(f"Eigenvalues of L^T * L: {eigvals}")
        print("==========================================\n")



    # def compute_s_from_feature(self, feature, task_id):
    #     s_cvx = cp.Variable(self.dim_latent)
    #     objective = cp.Minimize(
    #         0.5 * cp.sum_squares(feature - self.D @ s_cvx) + self.mu_one * cp.norm(s_cvx, 1)
    #     )
    #     prob = cp.Problem(objective)
    #     prob.solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6, max_iter=10000)
    #
    #     if s_cvx.value is not None:
    #         self.dict_s[task_id] = s_cvx.value.reshape(-1, 1)
    #     else:
    #         print("Lasso solver failed, s not updated.")

    def compute_s_from_feature(self, feature_low_dim, task_id):
        """
        feature_low_dim: numpy array, shape = (32,)
        task_id:      integer
        """
        # -- 0. For a new task without a previous s, initialize s=0 and skip solving -- #
        if task_id not in self.dict_s or np.allclose(self.dict_s[task_id], 0):
            # Initialize s as an all-zero column vector
            self.dict_s[task_id] = np.zeros((self.dim_latent, 1), dtype=np.float32)
            return

        # -- 1. Set up the CVXPY variable and objective -- #
        s_cvx = cp.Variable(self.dim_latent)

        objective = cp.Minimize(
            0.5 * cp.sum_squares(feature_low_dim - self.D @ s_cvx)
            + self.mu_one * cp.norm(s_cvx, 1)
        )
        prob = cp.Problem(objective)

        # -- 2. Warm start: use the previous s as the initial value, if any -- #
        s_initial = self.dict_s.get(task_id, None)
        if s_initial is not None:
            # s_initial has shape (K,1) -> flatten to (K,)
            s_cvx.value = s_initial.flatten()

        # -- 3. Solve the Lasso problem -- #
        try:
            prob.solve(solver=cp.OSQP, warm_start=True,
                       eps_abs=1e-6, eps_rel=1e-6, max_iters=20000)
        except cp.SolverError:
            # Fall back to SCS if OSQP fails
            try:
                prob.solve(solver=cp.SCS, verbose=False, eps=1e-6, max_iters=20000)
            except cp.SolverError:
                print(f"Lasso solver completely failed for task {task_id}.")
                return

        # -- 4. Check and update s -- #
        if s_cvx.value is not None:
            # Make sure s has shape (K,1)
            s_sol = s_cvx.value.reshape(-1, 1).astype(np.float32)
            self.dict_s[task_id] = s_sol
        else:
            print(f"Lasso solver failed to find a value for s (task {task_id}).")

    def update_dictionary_D(self, mu=1e-3):
        s_mat = np.hstack([self.dict_s[tid] for tid in sorted(self.task_feature_dict.keys())])
        feature_mat = np.hstack(
            [self.task_feature_dict[tid].reshape(-1, 1) for tid in sorted(self.task_feature_dict.keys())])

        for iteration in range(10):
            for j in range(self.dim_latent):
                s_j = s_mat[j, :].reshape(1, -1)
                residual = feature_mat - self.D @ s_mat + self.D[:, j:j + 1] @ s_j
                self.D[:, j] = (residual @ s_j.T / (np.linalg.norm(s_j) ** 2 + mu)).flatten()

        # def determine_task_id(self, server_feature_avg, server_id, threshold=0.98):
        #     if not self.task_feature_dict:
        #         task_id = 0
        #         self.task_feature_dict[task_id] = server_feature_avg
        #         self.task_server_mapping[task_id] = [server_id]
        #         print(f"[NEW TASK] Task 0 created with initial feature (Server {server_id})")
        #         return task_id, True  # new task
        # 
        #     max_sim = -1
        #     matched_task_id = None
        #     for tid, feature in self.task_feature_dict.items():
        #         sim = np.dot(server_feature_avg, feature) / (np.linalg.norm(server_feature_avg) * np.linalg.norm(feature))
        #         if sim > max_sim:
        #             max_sim = sim
        #             matched_task_id = tid
        # 
        #     # Print the maximum-similarity information
        #     print(f"[TASK SIMILARITY] Server {server_id}: Max similarity = {max_sim:.4f} (Matched task: {matched_task_id})")
        # 
        #     if max_sim >= threshold:
        #         # Update the mean feature
        #         self.task_feature_dict[matched_task_id] = 0.9 * self.task_feature_dict[
        #             matched_task_id] + 0.1 * server_feature_avg
        #         if server_id not in self.task_server_mapping[matched_task_id]:
        #             self.task_server_mapping[matched_task_id].append(server_id)
        #         return matched_task_id, False  # existing task
        #     else:
        #         # New task
        #         new_task_id = len(self.task_feature_dict)
        #         self.task_feature_dict[new_task_id] = server_feature_avg
        #         self.task_server_mapping[new_task_id] = [server_id]
        #         print(f"[NEW TASK] Task {new_task_id} created (Server {server_id}) with similarity {max_sim:.4f}")
        #         return new_task_id, True
        
# # Maintain matrix A_D and vector b_D as new attributes in __init__ (PG-ELLA dictionary)
# self.A_D = np.zeros((self.dim_latent * feature_dim, self.dim_latent * feature_dim))
# self.b_D = np.zeros((self.dim_latent * feature_dim, 1))
# self.mu_D = 1e-3
#
#
# # Updated in the same way as update_L:
# def update_dictionary_D_closed_form(self):
#     T = len(self.task_feature_dict)
#     feature_dim = next(iter(self.task_feature_dict.values())).shape[0]
#     I_D = np.eye(self.dim_latent * feature_dim) * self.mu_D
#
#     # Closed-form solution
#     vec_D = np.linalg.solve((1 / T) * self.A_D + I_D, (1 / T) * self.b_D)
#     self.D = vec_D.reshape(feature_dim, self.dim_latent)
#
#
# # Maintain A_D, b_D on every task update:
# def add_new_contribution_to_D(self, feature_new, s_new):
#     s_outer = s_new @ s_new.T
#     A_D_contrib = np.kron(s_outer, np.eye(len(feature_new)))
#     self.A_D += A_D_contrib
#     b_D_contrib = vec(np.kron(s_new.T, feature_new.T))
#     self.b_D += b_D_contrib

    def determine_task_id(self, server_feature_avg, server_id, threshold=0.85):
        """
        Lifelong task detection: decide which task server_feature_avg (raw feature vector) belongs to:
          - first L2-normalize the vector
          - if there is no task prototype yet, insert it as the first task
          - otherwise compute its cosine similarity to all prototypes and keep the maximum best_sim
          - if best_sim >= threshold it belongs to best_tid; update that cluster's sample pool and recompute its center
          - otherwise insert it as a new task
        Returns:
          task_id: integer task index
          is_new: bool, True if a new task was created
        """
        # -- 1. L2-normalize the input vector -- #
        # -- 1. Dimensionality-reduction preprocessing of the raw feature -- #
        # fa = self.preprocess_feature(server_feature_avg)  # unit vector after PCA

        norm_fa = np.linalg.norm(server_feature_avg)
        if norm_fa < 1e-8:
            # If (almost) all zeros, set it to zero
            fa = server_feature_avg * 0.0
        else:
            fa = server_feature_avg / norm_fa

        # -- 2. If there is no task prototype, create task_id=0 -- #
        if not self.task_feature_dict:
            tid = 0
            # Use the normalized vector as the first prototype center
            self.task_feature_dict[tid] = fa.copy()
            # Initialize this task's sample pool with this vector only
            self.task_feature_samples[tid] = [fa.copy()]
            # Record the servers belonging to this task
            self.task_server_mapping[tid] = [server_id]
            print(f"[NEW TASK] Task {tid} created with Server {server_id} as prototype.")
            return tid, True

        # -- 3. Cosine similarity to all prototypes, keep the maximum -- #
        sims = []
        for tid, feat_center in self.task_feature_dict.items():
            # feat_center is already a unit vector
            sim = float(np.dot(fa, feat_center))
            sims.append((tid, sim))

        # Sort similarities in descending order and take the first
        sims.sort(key=lambda x: x[1], reverse=True)
        best_tid, best_sim = sims[0]

        print(f"[TASK SIMILARITY] Server {server_id}: best_tid={best_tid}, best_sim={best_sim:.4f}")

        # -- 4. If the max similarity >= threshold, assign it to cluster best_tid -- #
        if best_sim >= threshold:
            # Add the vector to best_tid's sample pool
            samples = self.task_feature_samples[best_tid]
            samples.append(fa.copy())
            # Drop the oldest sample if the pool exceeds its maximum size
            if len(samples) > self.max_samples:
                samples.pop(0)
            # Recompute the cluster center as the mean of the pool
            new_center = np.mean(np.stack(samples, axis=0), axis=0)
            self.task_feature_dict[best_tid] = new_center / (np.linalg.norm(new_center) + 1e-8)

            # Record that this server belongs to the task cluster
            if server_id not in self.task_server_mapping[best_tid]:
                self.task_server_mapping[best_tid].append(server_id)
            return best_tid, False
        else:
            # -- 5. Otherwise create a new task prototype -- #
            new_tid = len(self.task_feature_dict)
            self.task_feature_dict[new_tid] = fa.copy()
            self.task_feature_samples[new_tid] = [fa.copy()]
            self.task_server_mapping[new_tid] = [server_id]
            print(f"[NEW TASK] Task {new_tid} created (Server {server_id}), best_sim={best_sim:.4f}")
            return new_tid, True

    def preprocess_feature(self, server_feature_avg):
        """
        Project an online raw feature with the fitted scaler + PCA and L2-normalize it, so cosine similarity can be used directly.
        server_feature_avg: numpy array, shape (203,)
        Returns: feat_unit: numpy array of shape (n_pca_components,)
        """
        # 1. Make sure the input is 2-D, shape (1, dim)
        feat203 = server_feature_avg.reshape(1, -1)  # -> (1, 203)
        # 2. Standardize
        feat_norm = self.scaler.transform(feat203)   # -> (1, 203)
        # 3. PCA projection
        feat_low = self.pca.transform(feat_norm).flatten()  # -> (32,)
        # 4. L2-normalize to a unit vector
        norm_low = np.linalg.norm(feat_low)
        if norm_low < 1e-8:
            return np.zeros_like(feat_low)
        return feat_low / norm_low
