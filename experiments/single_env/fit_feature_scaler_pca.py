"""Offline fit of the StandardScaler + PCA used by DC-ELLA's task detection.

Produces feature_scaler.pkl and feature_pca.pkl from the server feature
histories of an earlier pretraining/test run. The fitted files used in the
paper are already included, so this script is provided for provenance only;
point pretrain_path / test_path to your own run before using it.

Paper: "Deep Lifelong Learning for Adaptive Semantic-Aware Content Reuse in
UAV-Assisted Metaverse", IEEE Transactions on Mobile Computing.
Equation / section numbers refer to the paper.
"""

import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# --- 1. Load the pretraining and test data bundles ---
pretrain_path = "32_085_no_use/Hit+Similarity_pretrain_data.pkl"
test_path     = "32_085_no_use/Hit+Similarity_test_data.pkl"

with open(pretrain_path, "rb") as f:
    pretrain_data = pickle.load(f)
with open(test_path, "rb") as f:
    test_data = pickle.load(f)

# --- 2. Extract the server_feature_history dicts --- #
# pre_hist is a dict: {server_id: [array(203,), array(203,), ...], ...}
pre_hist_dict = pretrain_data["extra_stats"]["server_feature_history"]
test_hist_dict = test_data["extra_stats"]["server_feature_history"]

# Make sure both are dicts
if not isinstance(pre_hist_dict, dict) or not isinstance(test_hist_dict, dict):
    raise RuntimeError("server_feature_history should be a dict; check the data structure.")

# Sorted server ids for a consistent order
server_ids_pre = sorted(pre_hist_dict.keys())
server_ids_test = sorted(test_hist_dict.keys())

# Both sets of servers must match (otherwise later steps break)
if server_ids_pre != server_ids_test:
    raise RuntimeError("Server ids differ between pretraining and test data; please check.")
server_ids = server_ids_pre

n_servers = len(server_ids)
if n_servers == 0:
    raise RuntimeError("No servers found; cannot run PCA.")

# --- 3. Check record counts per server and align to the minimum --- #
lengths = [len(pre_hist_dict[sid]) for sid in server_ids]
lengths_test = [len(test_hist_dict[sid]) for sid in server_ids]

print("Pretraining records per server:", lengths)
print("Test records per server:", lengths_test)

# Minimum number of steps after alignment
n_steps = min(lengths)
n_steps_test = min(lengths_test)
dim = None

# Confirm each record is 1-D (e.g. shape=(203,)) and record dim
for sid in server_ids:
    hist0 = pre_hist_dict[sid][0]
    if not isinstance(hist0, np.ndarray) or hist0.ndim != 1:
        raise RuntimeError(f"Features of server {sid} are not a 1-D numpy array.")
    dim = hist0.shape[0]
    break

# --- 4. Convert the dicts to arrays of shape (n_servers, n_steps, dim) --- #
pre_array_list = []
for sid in server_ids:
    hist_list = pre_hist_dict[sid]
    if len(hist_list) < n_steps:
        raise RuntimeError(f"Server {sid} has fewer than {n_steps} pretraining records.")
    # Keep the first n_steps records and stack into (n_steps, dim)
    arr_j = np.stack(hist_list[:n_steps], axis=0)  # shape = (n_steps, dim)
    pre_array_list.append(arr_j)
# pre_array has shape (n_servers, n_steps, dim)
pre_array = np.stack(pre_array_list, axis=0)

test_array_list = []
for sid in server_ids:
    hist_list = test_hist_dict[sid]
    if len(hist_list) < n_steps_test:
        raise RuntimeError(f"Server {sid} has fewer than {n_steps_test} test records.")
    arr_j = np.stack(hist_list[:n_steps_test], axis=0)  # shape = (n_steps_test, dim)
    test_array_list.append(arr_j)
test_array = np.stack(test_array_list, axis=0)

print("pre_array.shape =", pre_array.shape)   # expected (n_servers, n_steps, dim)
print("test_array.shape =", test_array.shape) # expected (n_servers, n_steps_test, dim)

# --- 5. Transpose to (n_steps, n_servers, dim) and reshape to (n_steps*n_servers, dim) --- #
pre_feats_t  = pre_array.transpose(1, 0, 2)   # shape = (n_steps, n_servers, dim)
test_feats_t = test_array.transpose(1, 0, 2)  # shape = (n_steps_test, n_servers, dim)

pre_reshaped  = pre_feats_t.reshape(n_steps * n_servers, dim)
test_reshaped = test_feats_t.reshape(n_steps_test * n_servers, dim)

# --- 6. Concatenate and standardize --- #
all_feats = np.vstack([pre_reshaped, test_reshaped])  # shape = ((n_steps+n_steps_test)*n_servers, dim)
scaler = StandardScaler()
all_feats_norm = scaler.fit_transform(all_feats)

# --- 7. PCA dimensionality reduction (n_components below) --- #
n_components = 64
pca = PCA(n_components=n_components)
all_feats_pca = pca.fit_transform(all_feats_norm)

cum_var = np.cumsum(pca.explained_variance_ratio_)
print("PCA cumulative explained variance (first 64 dims):", cum_var[-1])

# --- 8. Split back into (n_steps, n_servers, k) and (n_steps_test, n_servers, k) --- #
pre_pca = all_feats_pca[: n_steps * n_servers].reshape(n_steps, n_servers, n_components)
test_pca = all_feats_pca[n_steps * n_servers : ].reshape(n_steps_test, n_servers, n_components)

# --- 9. Store the results in the original data structure ---
pretrain_data["extra_stats"]["server_feature_pca"] = pre_pca
test_data["extra_stats"]["server_feature_pca"]     = test_pca

# --- 10. Serialize the scaler and PCA model for online use --- #
joblib.dump(scaler, "feature_scaler.pkl")
joblib.dump(pca,    "feature_pca.pkl")

print("PCA done; results stored in extra_stats['server_feature_pca'].")
