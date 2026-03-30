# %% [markdown]
# # Federated Learning Sensitivity Analysis with Knowledge Distillation
# 
# This notebook presents a comprehensive analysis of **Federated Learning (FL)** under varying client configurations, with a focus on **client count sensitivity** and **data heterogeneity (IID vs non-IID)** for a multi-label classification task.
# 
# In addition to standard FL baselines, we introduce and evaluate **Knowledge Distillation (KD)** strategies designed to improve robustness under non-IID settings.
# 
# ### Core Components
# 
# - **Federated Training**  
#   Clients train models locally and synchronize via weighted aggregation (FedAvg). Variants such as **FedProx** are included to mitigate client drift.
# 
# - **Client Sensitivity Analysis**  
#   We systematically vary the number of clients (e.g., 2 → 10+) to study how increasing data fragmentation impacts performance under:
#   - IID splits
#   - Non-IID splits (label skew + size imbalance)
# 
# - **Knowledge Distillation (KD) Methods**
#   - **Option 1 (Local KD / Drift Control):**  
#     Each client distills knowledge from a frozen copy of the global model during local training.
#   - **Option 3 (Local Teacher → Global Student):**  
#     Each client maintains a stronger local teacher model and distills its knowledge into a shared global student.
# 
# - **Evaluation Metrics**  
#   Models are evaluated using:
#   - Macro / Micro F1-score
#   - AUC and PR-AUC
#   - Loss and prediction statistics
# 
# ### Goal
# 
# The objective is to understand how **federated optimization behaves under increasing client fragmentation and heterogeneity**, and whether **knowledge distillation can improve stability and generalization** in these settings.
# 
# ---

# %% [markdown]
# ## Imports

# %%
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from gensim.models    import Word2Vec
import torch.nn.functional as F
import torch.nn as nn
import torch

import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm
from evaluation import all_metrics

import math
import json
import os
import copy
import pandas as pd    # NEW – to store experiment results
import time             # NEW – to track runtime for each config
import itertools

# Optional: to ensure reproducibility
torch.manual_seed(42)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# %% [markdown]
# ## Data Loading and JSON Utilities
# 
# This section defines:
# - `load_data()` — loads tensors from disk (`../Data/X_type.pt`, `../Data/Y_type.pt`)  
#   and returns a PyTorch `DataLoader` for the chosen split.
# - `save_json()` and `load_json()` — simple JSON I/O helpers for saving and loading experiment logs.
# 

# %%
# Load Data

def load_data(split: str) -> DataLoader:
    """
    Load preprocessed tensor data for a given split.

    Args:
        split (str): One of {'train', 'val', 'test'}.

    Returns:
        DataLoader: A DataLoader wrapping the corresponding dataset.
    """
    X_data = torch.load(os.path.join("..", "Data", f"X_{split}.pt"))
    Y_data = torch.load(os.path.join("..", "Data", f"Y_{split}.pt"))

    return DataLoader(
        TensorDataset(X_data, Y_data),
        batch_size=32,
        shuffle=False,
        pin_memory=True
    )

# %%
# JSON I/O Utils

def save_json(data: dict, filepath: str) -> None:
    """
    Save a Python dictionary to a JSON file.

    Args:
        data (dict): Data to be saved.
        filepath (str): Destination file path.
    """
    with open(filepath, mode="w+") as f:
        json.dump(data, fp=f, indent=2)


def load_json(filepath: str) -> dict:
    """
    Load JSON data from a file.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        dict: Loaded data.
    """
    with open(filepath, mode="r") as f:
        return json.load(f)

# %% [markdown]
# ## Model Definition — ConvAttnPool
# 
# This section defines the **ConvAttnPool** model, which combines:
# - **Convolutional layers** for feature extraction,
# - **Attention pooling** to capture weighted feature importance,
# - And a **final classifier** for binary prediction.
# 
# A key modification (as noted earlier) is the inclusion of the **embedding table** within the model itself for modularity.
# 

# %%
# Model Architecture

class ConvAttnPool(nn.Module):
    """
    Convolution + Attention Pooling model using a pretrained Word2Vec embedding table.

    Args:
        table_path (str): Path to the pretrained Word2Vec model (.w2v file).
        label_space (int): Number of output labels/classes.
        num_of_filters (int): Number of convolutional filters.
        kernel_size (int): Kernel size for the Conv1d layer.
        drop_out (float): Dropout probability.

    Attributes:
        embed (nn.Embedding): Embedding layer initialized from pretrained vectors.
        conv (nn.Conv1d): Convolutional feature extractor.
        U (nn.Linear): Linear layer for attention projection.
        final (nn.Linear): Linear layer for classification weights.
        embed_drop (nn.Dropout): Dropout applied after embeddings.
    """

    def __init__(self, table_path: str, label_space: int = 50, num_of_filters: int = 10, kernel_size: int = 3, drop_out: float = 0.2):
        super().__init__()

        # Load pretrained Word2Vec model
        model = Word2Vec.load(table_path)
        vocab_size, embed_d = model.wv.vectors.shape

        # Prepare embedding table (append a zero vector for padding index)
        embed_table = torch.from_numpy(model.wv.vectors).float()
        embed_table = torch.cat([embed_table, torch.zeros((1, embed_d))], dim=0)

        # Embedding layer
        self.embed = nn.Embedding.from_pretrained(embeddings=embed_table, padding_idx=vocab_size)
        self.embed_drop = nn.Dropout(p=drop_out)

        # Convolutional feature extractor
        self.conv = nn.Conv1d(
            in_channels=embed_d,
            out_channels=num_of_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        # Attention and output layers
        self.U = nn.Linear(num_of_filters, label_space)
        self.final = nn.Linear(num_of_filters, label_space)

        # Store embedding dimension for reference
        self.embedding_size = embed_d

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of token indices with shape (batch_size, seq_len).

        Returns:
            tuple:
                y (torch.Tensor): Logits for each label (batch_size, label_space).
                alpha (torch.Tensor): Attention weights (batch_size, label_space, seq_len).
        """
        x = self.embed(x)                # (B, L, embed_d)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)            # (B, embed_d, L)
        x = torch.tanh(self.conv(x).transpose(1, 2))  # (B, L, num_of_filters)

        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)  # (B, label_space, L)
        m = alpha.matmul(x)             # (B, label_space, num_of_filters)
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)       # (B, label_space)

        return y, alpha

# %%
# Model Factory

def GenerateModel(table_path: str, num_of_filters: int = 15, kernel_size: int = 5) -> ConvAttnPool:
    """
    Factory function to create a ConvAttnPool model with standard hyperparameters.

    Args:
        table_path (str): Path to the pretrained Word2Vec model.
        num_of_filters (int): Number of convolutional filters.
        kernel_size (int): Kernel size for Conv1d.

    Returns:
        ConvAttnPool: Initialized model instance.
    """
    return ConvAttnPool(
        table_path=table_path,
        drop_out=0.2,
        num_of_filters=num_of_filters,
        label_space=50,
        kernel_size=kernel_size
    )


# %% [markdown]
# ## Federated Learning Components
# 
# This section defines the two core routines of the federated learning process:
# 
# 1. **`FedAvg`** — performs *federated averaging* by combining model weights from multiple clients into a single global model.
# 2. **`client_update`** — trains a model locally on one client’s data for a fixed number of epochs.
# 
# Together, they form the backbone of the **federated training loop**, where multiple clients train in parallel and periodically synchronize with the global model.
# 

# %% [markdown]
# ### Federated Averaging (Parameter Dictionary Form)
# 
# This version of **FedAvg** operates directly on dictionaries of tensors rather than full model objects.
# 
# Each client provides a dictionary of parameters (e.g., layer weights).  
# The function stacks corresponding parameters across clients and computes their element-wise mean to update the global parameters.
# 
# This approach:
# - Avoids unnecessary deep copies of entire models.
# - Keeps aggregation efficient and transparent.
# 

# %%
# FedAvg - working with parameter dictionary rather than deepcopy

def FedAvg(global_model: dict, client_state_dicts: list[dict]) -> dict:
    """
    Perform Federated Averaging (FedAvg) on parameter dictionaries.

    Args:
        global_model (dict): Global model parameter dictionary (in-place update).
        client_state_dicts (list[dict]): List of parameter dictionaries from clients.

    Returns:
        dict: Updated global parameter dictionary (averaged across clients).
    """
    for key in global_model.keys():
        # Stack corresponding parameters from all clients and take mean
        stacked = torch.stack(
            [client_dict[key].float() for client_dict in client_state_dicts],
            dim=0
        )
        global_model[key] = torch.mean(stacked, dim=0)
    return global_model

# %%
# --- FedProx and SCAFFOLD Aggregation Methods ---

def FedProx(global_model_dict, client_state_dicts, mu=0.01):
    """
    FedProx aggregation (same averaging as FedAvg, 
    since proximal regularization happens in local training).
    
    Args:
        global_model_dict (dict): Global model parameters.
        client_state_dicts (list[dict]): List of client parameter dicts.
        mu (float): Proximal term weight (applied during local updates).
    """
    # FedProx uses FedAvg-style aggregation; proximal term affects client training only.
    return FedAvg(global_model_dict, client_state_dicts)


def Scaffold(global_model_dict, client_state_dicts, c_global, c_clients, lr, num_clients):
    """
    SCAFFOLD server update rule:
        w_{t+1} = w_t + (1/K) * Σ [Δw_k - lr * (c_k - c)]
    
    Args:
        global_model_dict: current global weights (dict of tensors)
        client_state_dicts: list of client state_dicts after local updates
        c_global: global control variate dict
        c_clients: list of local control variate dicts
        lr: learning rate
        num_clients: number of clients participating this round
    
    Returns:
        Updated (global_model_dict, c_global, c_clients)
    """
    new_global = copy.deepcopy(global_model_dict)

    # Average model deltas with control variate correction
    for key in global_model_dict.keys():
        # Δw_k = w_k - w_global
        deltas = torch.stack(
            [client_state_dicts[k][key] - global_model_dict[key] for k in range(num_clients)],
            dim=0
        )
        mean_delta = torch.mean(deltas, dim=0)

        # correction term from c_k - c
        correction = torch.stack(
            [c_clients[k][key] - c_global[key] for k in range(num_clients)],
            dim=0
        ).mean(dim=0)

        # apply update
        new_global[key] = global_model_dict[key] + mean_delta - lr * correction

    # update global control variate
    for key in c_global.keys():
        delta_cs = torch.stack(
            [c_clients[k][key] - c_global[key] for k in range(num_clients)],
            dim=0
        )
        c_global[key] = c_global[key] + (1 / num_clients) * delta_cs.mean(dim=0)

    return new_global, c_global, c_clients


# %% [markdown]
# ### Client Update Routine
# 
# Each client performs local training on its own dataset for a fixed number of epochs.  
# After training, the function returns:
# - The **final local loss** for logging.
# - The **updated model parameters** (`state_dict`) to be sent back to the server.
# 
# This implementation uses:
# - **Adam optimizer** with β = (0.9, 0.99)
# - **Binary Cross-Entropy with Logits** loss (`BCEWithLogitsLoss`)
# 

# %%
# fix multi label collapsing to all 0s problem by having positive class weighting
def compute_pos_weight(train_loader, n_labels):
    pos = torch.zeros(n_labels)
    total = 0
    for _, y in train_loader:
        pos += y.sum(dim=0)
        total += y.shape[0]
    neg = total - pos
    return (neg / pos.clamp_min(1.0)).float()

# %%
# Create focal loss function

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_term = (1 - pt).pow(self.gamma)

        if self.alpha is not None:
            alpha_term = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_term = alpha_term * focal_term

        loss = focal_term * bce_loss
        return loss.mean() if self.reduction == "mean" else loss.sum()


# %%
def client_update(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 1,
    lr: float = 0.1,
    device: str = "cpu",
    use_focal: bool = False,
    gamma: float = 2.5,
    mu: float = 0.01,                   # FedProx proximal coefficient
    global_params: dict = None,         # for FedProx / SCAFFOLD
    c_global: dict = None,              # for SCAFFOLD
    c_local: dict = None,               # for SCAFFOLD
    algorithm: str = "FedAvg"           # which algorithm is being used
) -> tuple[float, dict, dict]:
    """
    Perform local training for a single client.
    Supports FedAvg, FedProx, and SCAFFOLD.
    Returns (final_loss, updated_model_state, updated_c_local)
    """
    model.to(device)
    model.train()

    n_labels = train_loader.dataset[0][1].shape[0]
    pos_weight = compute_pos_weight(train_loader, n_labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

    if use_focal:
        alpha = torch.clamp(pos_weight / pos_weight.max(), min=0.1, max=0.9).to(device)
        loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Training loop ---
    for _ in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds, _ = model(X_batch)
            loss = loss_fn(preds, y_batch)

            # --- FedProx proximal term ---
            if algorithm == "FedProx" and global_params is not None:
                prox_term = 0.0
                for w, w_global in zip(model.parameters(), global_params.values()):
                    prox_term += (w - w_global.to(device)).norm(2) ** 2
                loss += (mu / 2) * prox_term

            optimizer.zero_grad()
            loss.backward()

            # --- SCAFFOLD correction ---
            if algorithm == "SCAFFOLD" and c_global is not None and c_local is not None:
                with torch.no_grad():
                    for w, cg, cl in zip(model.parameters(), c_global.values(), c_local.values()):
                        if w.grad is not None:
                            w.grad -= (cg.to(device) - cl.to(device))

            optimizer.step()

    return loss.item(), model.state_dict(), c_local


# %% [markdown]
# ## Federated Training — Full Experiment Pipeline
# 
# This section coordinates the **federated learning process**:
# 1. Initializes global and client models.
# 2. Splits the dataset into client partitions.
# 3. Iteratively performs:
#    - Local training (`client_update`)
#    - Model aggregation (`FedAvg`)
#    - Periodic evaluation and checkpointing
# 
# Metrics are saved incrementally to `../History/logs/metric_history.json`, and the best models (by AUC and F1) are checkpointed.
# 

# %% [markdown]
# ### Set up

# %%
# Config

config = {
    "batch_size": 32,
    "lr": 0.002,
    "n_filters": 21,
    "window_size": 6,
    "epochs": 3,             # default (overridden per experiment)
    "rounds": 10,            # communication rounds per experiment
    "use_focal": False,      
    "gamma": 2.5,            # focal loss focusing parameter
    "mu": 0.01,              # FedProx proximal term coefficient
    "algorithm": "FedAvg"    # will be updated in loop to FedAvg, FedProx, or SCAFFOLD
}

# Path to pretrained embedding table
model_param_path = os.path.join("..", "Model", "processed_full.w2v")

# %%
# Load full training dataset (clients will be split dynamically later)
X_train = torch.load(os.path.join("..", "Data", "X_train.pt"))
Y_train = torch.load(os.path.join("..", "Data", "Y_train.pt"))
train_dataset = TensorDataset(X_train, Y_train)

print(f"Loaded full training dataset: {len(train_dataset)} samples.")

val_loader = load_data("val")
test_loader = load_data("test")

# %% [markdown]
# ### Eval stuff

# %%
# Auto tuning to find best global threshold

@torch.no_grad()
def find_best_threshold(model: nn.Module, data_loader: DataLoader, device: torch.device):
    """
    Sweeps multiple thresholds on the validation set to find the one 
    that maximizes F1_micro.

    Returns:
        tuple (best_f1, best_threshold)
    """
    model.eval()
    all_pred_raw = torch.empty(0, dtype=torch.float32, device=device)
    all_labels = torch.empty(0, dtype=torch.float32, device=device)

    for X_batch, y_batch in data_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds, _ = model(X_batch)
        all_pred_raw = torch.cat([all_pred_raw, preds], dim=0)
        all_labels = torch.cat([all_labels, y_batch], dim=0)

    best_f1, best_thr = 0.0, 0.1
    for t in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        preds_t = (torch.sigmoid(all_pred_raw) >= t).long()
        m = all_metrics(
            yhat=preds_t.cpu().numpy(),
            y=all_labels.cpu().numpy(),
            yhat_raw=all_pred_raw.cpu().numpy()
        )
        if m["f1_micro"] > best_f1:
            best_f1, best_thr = m["f1_micro"], t

    return best_f1, best_thr


# %%
# Tune to find best threshold per label

@torch.no_grad()
def find_best_thresholds_per_label(model: nn.Module, data_loader: DataLoader, device: torch.device):
    """
    Finds an optimal sigmoid threshold per label to maximize F1 for each label independently.

    Returns:
        tuple:
            - macro_f1 (float): Average of best per-label F1s
            - thresholds (Tensor): Shape (num_labels,) with best threshold per label
    """
    model.eval()
    all_pred_raw, all_labels = [], []
    for X_batch, y_batch in data_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds, _ = model(X_batch)
        all_pred_raw.append(preds)
        all_labels.append(y_batch)

    all_pred_raw = torch.cat(all_pred_raw)
    all_labels = torch.cat(all_labels)
    sigm = torch.sigmoid(all_pred_raw)

    n_labels = all_labels.shape[1]
    best_thresholds = torch.zeros(n_labels, device=device)
    best_f1s = torch.zeros(n_labels, device=device)

    for i in range(n_labels):
        best_f, best_t = 0.0, 0.3
        for t in torch.arange(0.05, 0.95, 0.05):
            preds_i = (sigm[:, i] >= t).long()
            y_i = all_labels[:, i].long()
            tp = (preds_i * y_i).sum().item()
            fp = (preds_i * (1 - y_i)).sum().item()
            fn = ((1 - preds_i) * y_i).sum().item()
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            f1 = 2 * prec * rec / (prec + rec + 1e-9)
            if f1 > best_f:
                best_f, best_t = f1, t
        best_thresholds[i] = best_t
        best_f1s[i] = best_f

    macro_f1 = best_f1s.mean().item()
    print(f"[Per-Label Thresholds] Macro F1={macro_f1:.4f}")
    return macro_f1, best_thresholds.cpu()

# %%

def _fmt(x):
    """Safely format floats that might be None or NaN."""
    if x is None:
        return "n/a"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "n/a"
    return f"{x:.4f}"

@torch.no_grad()
def eval_model(
    model: nn.Module,
    device: torch.device,
    data_loader: DataLoader,
    tune_threshold=False,
    fixed_thr=0.3,
    sigmoid=False,
    per_label_thr=None
):
    model.eval()
    model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    all_pred_raw = torch.empty(0, dtype=torch.float32, device=device)
    all_labels = torch.empty(0, dtype=torch.float32, device=device)
    total_loss = 0.0

    for X_batch, y_batch in data_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds, _ = model(X_batch)
        loss = loss_fn(preds, y_batch)
        total_loss += loss.item()
        all_pred_raw = torch.cat([all_pred_raw, preds], dim=0)
        all_labels = torch.cat([all_labels, y_batch], dim=0)

    avg_loss = total_loss / len(data_loader)

    sigmoid_vals = torch.sigmoid(all_pred_raw)
    if per_label_thr is not None:
        pred_labels = (sigmoid_vals >= per_label_thr.to(device)).long()
        best_f1, best_thr = None, "per-label"
    else:
        pred_labels = (sigmoid_vals >= fixed_thr).long()
        metrics = all_metrics(
            yhat=pred_labels.cpu().numpy(),
            y=all_labels.cpu().numpy(),
            yhat_raw=all_pred_raw.cpu().numpy()
        )
        best_f1, best_thr = metrics["f1_micro"], fixed_thr
        if tune_threshold:
            best_f1, best_thr = find_best_threshold(model, data_loader, device)

    if per_label_thr is not None:
        metrics = all_metrics(
            yhat=pred_labels.cpu().numpy(),
            y=all_labels.cpu().numpy(),
            yhat_raw=all_pred_raw.cpu().numpy()
        )

    pr_macro = metrics.get("pr_auc_macro")
    pr_micro = metrics.get("pr_auc_micro")
    auc_macro = metrics.get("auc_macro")
    auc_micro = metrics.get("auc_micro")

    avg_pred_labels = pred_labels.sum(dim=1).float().mean().item()

    print(
        f"[Eval] Avg loss={_fmt(avg_loss)} | "
        f"F1_micro={_fmt(metrics.get('f1_micro'))} | F1_macro={_fmt(metrics.get('f1_macro'))} | "
        f"AUC_macro={_fmt(auc_macro)} | AUC_micro={_fmt(auc_micro)} | "
        f"PR-AUC_macro={_fmt(pr_macro)} | PR-AUC_micro={_fmt(pr_micro)} | "
        f"Best_F1={_fmt(best_f1 if best_f1 is not None else metrics.get('f1_micro'))} @ thr={best_thr} | "
        f"Avg labels/sample={avg_pred_labels:.2f}"
    )

    metrics["best_f1_micro"] = best_f1 if best_f1 else metrics["f1_micro"]
    metrics["best_thr"] = best_thr
    return avg_loss, metrics


# %% [markdown]
# ## Client Number Sensitivity Analysis
# 
# Goal: test how FL performance changes as the number of simulated clients increases.
# 
# **Option A (implemented now): IID split**
# - Randomly split the same training dataset into K approximately equal partitions.
# 
# **Option B (later): non-IID split**
# - Split data to mimic hospital heterogeneity (label prevalence shifts, size imbalance, etc.).
# 
# We keep everything else fixed (model, optimizer, rounds, local epochs) so the only change is
# the number of clients and the resulting partitioning.

# %%
# do weighted by partition size instead of mean by clients (important for different sizes like part b)

def FedAvg_weighted(client_state_dicts: list[dict], client_sizes: list[int]) -> dict:
    """
    Weighted FedAvg by client dataset size. Returns a NEW state_dict (does not mutate inputs).
    client_state_dicts: list of state_dicts (as produced by model.state_dict())
    client_sizes: list of ints (number of samples per client)
    """
    total = float(sum(client_sizes))
    out = {}
    # iterate keys from first client's state_dict
    for key in client_state_dicts[0].keys():
        # accumulate weighted sum
        acc = None
        for sd, n in zip(client_state_dicts, client_sizes):
            term = sd[key].float() * (n / total)
            acc = term if acc is None else acc + term
        out[key] = acc
    return out

# %% [markdown]
# ### IID Split

# %%
# IID split helper for option A

def split_dataset_iid(dataset, num_clients: int, seed: int = 42):
    """
    IID split: shuffle indices and split into K chunks.
    Returns list of Subset datasets.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)

    # chunk sizes (nearly equal)
    base = len(dataset) // num_clients
    rem  = len(dataset) % num_clients
    sizes = [base + (1 if i < rem else 0) for i in range(num_clients)]

    subsets = []
    start = 0
    for sz in sizes:
        subsets.append(Subset(dataset, indices[start:start+sz].tolist()))
        start += sz
    return subsets

# %% [markdown]
# ### Non-IID Split (Option B)
# 
# Multi-label non-IID split that simulates:
# - size imbalance across hospitals (Dirichlet)
# - per-hospital label-prevalence shifts (preferred labels + bias)
# 
# Parameters:
# - size_alpha: smaller -> more size imbalance
# - labels_per_client: how many "specialty" labels each client prefers
# - bias_strength: probability of sampling examples that contain preferred labels

# %%
# Non-IID splitter for multi-label datasets

def split_dataset_noniid_multilabel(
    dataset,
    num_clients: int,
    seed: int = 42,
    size_alpha: float = 0.5,          # Dirichlet parameter for client sizes
    labels_per_client: int = 10,      # number of preferred labels per client
    bias_strength: float = 0.85       # prob of sampling from preferred-label examples
):
    """
    Splits `dataset` (TensorDataset or similar with Y in dataset.tensors[1])
    into `num_clients` Subset objects with non-IID label prevalence and size imbalance.
    """
    rng = np.random.default_rng(seed)

    # extract Y matrix
    if hasattr(dataset, "tensors"):
        Y = dataset.tensors[1].cpu()
    else:
        # fallback for generic dataset: collect Y
        Y = torch.stack([dataset[i][1] for i in range(len(dataset))]).cpu()

    n = len(dataset)
    n_labels = Y.shape[1]

    # --- sizes via Dirichlet ---
    props = rng.dirichlet(alpha=np.ones(num_clients) * size_alpha)
    sizes = (props * n).astype(int)
    diff = n - sizes.sum()
    for i in range(abs(diff)):
        sizes[i % num_clients] += 1 if diff > 0 else -1
    sizes = sizes.tolist()

    # --- label -> indices map ---
    label_to_indices = defaultdict(list)
    for idx in range(n):
        labs = torch.nonzero(Y[idx]).flatten().tolist()
        for lab in labs:
            label_to_indices[lab].append(idx)

    # --- preferred labels per client ---
    all_labels = np.arange(n_labels)
    preferred = []
    for _ in range(num_clients):
        preferred.append(rng.choice(all_labels, size=min(labels_per_client, n_labels), replace=False).tolist())

    # --- assignment ---
    unassigned = set(range(n))
    client_indices = [[] for _ in range(num_clients)]

    def pick_preferred(client_id):
        labs = preferred[client_id]
        candidates = []
        for lab in labs:
            candidates.extend(label_to_indices.get(lab, []))
        if not candidates:
            return None
        rng.shuffle(candidates)
        for idx in candidates:
            if idx in unassigned:
                return idx
        return None

    # fill clients
    for cid in range(num_clients):
        target = sizes[cid]
        while len(client_indices[cid]) < target and unassigned:
            use_pref = rng.random() < bias_strength
            idx = pick_preferred(cid) if use_pref else None

            if idx is None:
                # choose uniformly from remaining
                idx = rng.choice(list(unassigned))

            client_indices[cid].append(int(idx))
            unassigned.remove(int(idx))

    # distribute any leftovers
    if unassigned:
        leftovers = list(unassigned)
        rng.shuffle(leftovers)
        for i, idx in enumerate(leftovers):
            client_indices[i % num_clients].append(int(idx))

    # build subsets
    subsets = [Subset(dataset, inds) for inds in client_indices]
    return subsets

# %% [markdown]
# ### Targeted Sensitivity Analysis

# %%
def run_fl_sensitivity_once(
    split_mode: str,            # "iid" or "noniid"
    num_clients: int,
    local_epochs: int,
    config: dict,
    train_dataset,
    val_loader,
    test_loader,
    device,
    seed: int = 42,
    algo: str = "FedAvg",        # "FedAvg" or "FedProx"
    mu: float = 0.01,            # only used for FedProx
    # non-IID params
    size_alpha: float = 0.5,
    labels_per_client: int = 10,
    bias_strength: float = 0.85
):
    start_time = time.time()

    # --- split ---
    if split_mode == "iid":
        client_datasets = split_dataset_iid(train_dataset, num_clients=num_clients, seed=seed)
    elif split_mode == "noniid":
        client_datasets = split_dataset_noniid_multilabel(
            train_dataset,
            num_clients=num_clients,
            seed=seed,
            size_alpha=size_alpha,
            labels_per_client=labels_per_client,
            bias_strength=bias_strength
        )
    else:
        raise ValueError("split_mode must be 'iid' or 'noniid'")

    client_sizes = [len(cd) for cd in client_datasets]
    client_loaders = [DataLoader(cd, batch_size=config["batch_size"], shuffle=True) for cd in client_datasets]

    # --- init global model ---
    global_model = GenerateModel(
        model_param_path,
        num_of_filters=config["n_filters"],
        kernel_size=config["window_size"]
    ).to(device)

    client_model = copy.deepcopy(global_model)

    # --- FL rounds ---
    for rnd in tqdm(range(config["rounds"]), colour="blue",
                    desc=f"{algo} {split_mode} | K={num_clients} | E={local_epochs}"):
        client_params = []
        for loader in client_loaders:
            client_model.load_state_dict(global_model.state_dict())

            _, client_state, _ = client_update(
                model=client_model,
                train_loader=loader,
                epochs=local_epochs,
                lr=config["lr"],
                device=device,
                use_focal=config["use_focal"],
                gamma=config["gamma"],
                mu=mu,
                global_params=(global_model.state_dict() if algo == "FedProx" else None),
                c_global=None,
                c_local=None,
                algorithm=algo,
            )
            client_params.append(client_state)

        # server aggregation: sample-size weighted FedAvg (safe for IID + nonIID)
        new_params = FedAvg_weighted(client_params, client_sizes)
        global_model.load_state_dict(new_params)

    # --- eval ---
    _, per_label_thr = find_best_thresholds_per_label(global_model, val_loader, device)
    _, metrics = eval_model(global_model, device, test_loader, per_label_thr=per_label_thr)

    elapsed = time.time() - start_time
    return metrics, elapsed

# %%
# # =========================
# # Sensitivity Sweep (FULL)
# # =========================

# client_counts = [2, 3, 5, 8, 10]      # 2–20 as requested

# # keep fast first; later bump rounds to your real value
# sweep_config = config.copy()
# sweep_config["rounds"] = 10  # debug; set higher later

# # choose non-IID severity (keep fixed across all methods)
# noniid_params = dict(size_alpha=0.5, labels_per_client=10, bias_strength=0.85)

# # methods to compare
# METHODS = [
#     #dict(method="FedAvg",  algo="FedAvg",  mu=None,  local_epochs=3)
#     dict(method="FedAvg_E1", algo="FedAvg", mu=None, local_epochs=1),

#     #dict(method="FedProx_mu0.01_E3", algo="FedProx", mu=0.01, local_epochs=3)
#     #dict(method="FedProx_mu0.1_E3",  algo="FedProx", mu=0.10, local_epochs=3)

#     dict(method="FedProx_mu0.01_E1", algo="FedProx", mu=0.01, local_epochs=1),
#     dict(method="FedProx_mu0.1_E1",  algo="FedProx", mu=0.10, local_epochs=1),

# ]

# out_path = "../History/sensitivity_all_methods.csv"

# all_rows = []
# for split_mode in ["iid", "noniid"]:
#     for m in METHODS:
#         for k in client_counts:
#             print(f"\n=== {split_mode.upper()} | {m['method']} | K={k} ===")

#             metrics, elapsed = run_fl_sensitivity_once(
#                 split_mode=split_mode,
#                 num_clients=k,
#                 local_epochs=m["local_epochs"],
#                 config=sweep_config,
#                 train_dataset=train_dataset,
#                 val_loader=val_loader,
#                 test_loader=test_loader,
#                 device=device,
#                 seed=42,
#                 algo=m["algo"],
#                 mu=(m["mu"] if m["mu"] is not None else 0.0),
#                 **(noniid_params if split_mode == "noniid" else {})
#             )

#             row = dict(
#                 split=split_mode,
#                 method=m["method"],
#                 algo=m["algo"],
#                 mu=m["mu"],
#                 local_epochs=m["local_epochs"],
#                 clients=k,
#                 auc_macro=metrics.get("auc_macro"),
#                 auc_micro=metrics.get("auc_micro"),
#                 f1_macro=metrics.get("f1_macro"),
#                 f1_micro=metrics.get("f1_micro"),
#                 pr_auc_macro=metrics.get("pr_auc_macro"),
#                 pr_auc_micro=metrics.get("pr_auc_micro"),
#                 time_sec=elapsed,
#             )
#             if split_mode == "noniid":
#                 row.update(noniid_params)

#             all_rows.append(row)

#             pd.DataFrame(all_rows).to_csv(out_path, index=False)

# print(f"\nDone. Saved to {out_path}")
# sens_all = pd.DataFrame(all_rows)
# display(sens_all)

# %%
# # Load + table (primary metrics)

# out_path = "../History/sensitivity_all_methods.csv"
# sens_all = pd.read_csv(out_path)
# print("Loaded:", out_path, "rows=", len(sens_all))

# # Comparison table: primary metrics (Macro F1, PR-AUC Macro)
# table = sens_all[["split","method","clients","f1_macro","pr_auc_macro","time_sec"]].copy()
# display(table.head())

# %%
# # Plotting

# # Pre-define visually distinct markers/linestyles (works for both color and B/W)
# MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
# LINESTYLES = ["-", "--", "-.", ":", (0,(3,1,1,1)), (0,(5,5)), (0,(1,1))]

# def plot_macro_f1_all_methods(df, split_mode: str, bw: bool = False):
#     d = df[df["split"] == split_mode].copy()

#     # consistent order in legend (so runs are stable)
#     methods = sorted(d["method"].unique())

#     plt.figure(figsize=(10, 5))

#     # cycle marker/style pairs so each method is distinguishable
#     style_cycle = itertools.cycle([
#         (m, ls) for m in MARKERS for ls in LINESTYLES
#     ])

#     for method in methods:
#         g = d[d["method"] == method].sort_values("clients")
#         marker, ls = next(style_cycle)

#         if bw:
#             plt.plot(
#                 g["clients"], g["f1_macro"],
#                 color="black",
#                 linestyle=ls,
#                 marker=marker,
#                 markersize=6,
#                 linewidth=2,
#                 label=method
#             )
#         else:
#             plt.plot(
#                 g["clients"], g["f1_macro"],
#                 linestyle=ls,
#                 marker=marker,
#                 markersize=6,
#                 linewidth=2,
#                 label=method
#             )

#     plt.xlabel("Number of Clients")
#     plt.ylabel("Macro F1")
#     plt.title(f"Macro F1 vs #Clients ({split_mode.upper()})")
#     plt.grid(True, alpha=0.3)
#     plt.legend(fontsize=8, ncols=2, frameon=True)
#     plt.tight_layout()
#     plt.show()

# # Color
# plot_macro_f1_all_methods(sens_all, "iid", bw=False)
# plot_macro_f1_all_methods(sens_all, "noniid", bw=False)

# # Black & White
# plot_macro_f1_all_methods(sens_all, "iid", bw=True)
# plot_macro_f1_all_methods(sens_all, "noniid", bw=True)

# %% [markdown]
# ## Knowledge Distillation Extensions (Option 1 and Option 3)
# 
# This section adds two KD-based federated training variants for the multi-label ICD task:
# 
# **Option 1 — Local KD / Drift Control**
# - Each client receives the current global model.
# - A frozen copy acts as the **teacher**.
# - A trainable copy acts as the **student**.
# - The student is trained on:
#   - supervised BCE loss on ground-truth labels
#   - KD loss against the frozen global teacher
# 
# **Option 3 — Local Heterogeneous Teacher → Global Student**
# - The server maintains a shared global **student** model.
# - Each client maintains its own persistent local **teacher**.
# - The teacher is trained locally on hard labels.
# - The student is trained on:
#   - supervised BCE loss
#   - KD loss against the local teacher
# 
# Notes:
# - These implementations are designed for **multi-label** prediction.
# - Server-side distillation / FedDKD is intentionally deferred to a later section.
# - FedProx compatibility is included by applying the proximal penalty to the uploaded student model.

# %%
# KD configuration defaults (can be adjusted later)

kd_config = {
    "kd_alpha": 0.5,          # weight on supervised loss
    "kd_temperature": 2.0,    # temperature for KD
    "teacher_filters": 32,    # larger local teacher for Option 3
    "teacher_window_size": 6
}

# %% [markdown]
# ### KD Loss Utilities
# 
# For this ICD setting, the task is **multi-label**, so we do not use softmax distillation.
# Instead, we match temperature-scaled sigmoid outputs label-wise.

# %%
def build_supervised_loss(
    train_loader,
    n_labels,
    use_focal: bool = False,
    gamma: float = 2.5,
    device: str = "cpu"
):
    """
    Build the supervised loss used for a client's local dataset.
    """
    pos_weight = compute_pos_weight(train_loader, n_labels).to(device)

    if use_focal:
        alpha = torch.clamp(pos_weight / pos_weight.max(), min=0.1, max=0.9).to(device)
        return FocalLoss(alpha=alpha, gamma=gamma)
    else:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def multilabel_kd_loss(student_logits, teacher_logits, temperature: float = 2.0):
    """
    Multi-label KD loss using temperature-scaled sigmoid probabilities.

    This behaves like a binary KL-style matching loss across labels.
    """
    T = temperature

    teacher_probs = torch.sigmoid(teacher_logits / T)
    student_probs = torch.sigmoid(student_logits / T)

    kd = F.binary_cross_entropy(
        student_probs,
        teacher_probs,
        reduction="mean"
    )

    return kd * (T * T)

# %% [markdown]
# ### Proximal Regularization Helper
# 
# Used when combining Option 1 or Option 3 with FedProx.
# The proximal penalty is applied only to the uploaded student model.

# %%
def proximal_term(model: nn.Module, global_params: dict, device: str = "cpu"):
    """
    Compute FedProx proximal penalty relative to the received global model.
    """
    prox = 0.0
    for name, param in model.named_parameters():
        prox += torch.norm(param - global_params[name].to(device), p=2) ** 2
    return prox

# %% [markdown]
# ### Option 1 — Local KD / Drift Control
# 
# Teacher: frozen copy of the global model  
# Student: trainable local copy initialized from the same global model

# %%
def client_update_option1(
    global_model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 1,
    lr: float = 0.001,
    device: str = "cpu",
    use_focal: bool = False,
    gamma: float = 2.5,
    kd_alpha: float = 0.5,
    kd_temperature: float = 2.0,
    use_fedprox: bool = False,
    mu: float = 0.01
):
    """
    Option 1:
    - teacher = frozen global model
    - student = local trainable model
    - local loss = supervised + KD (+ optional FedProx penalty)
    """
    teacher = copy.deepcopy(global_model).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = copy.deepcopy(global_model).to(device)
    student.train()

    global_params = {
        k: v.detach().clone()
        for k, v in global_model.state_dict().items()
    }

    n_labels = train_loader.dataset[0][1].shape[0]
    sup_loss_fn = build_supervised_loss(
        train_loader=train_loader,
        n_labels=n_labels,
        use_focal=use_focal,
        gamma=gamma,
        device=device
    )

    optimizer = torch.optim.Adam(student.parameters(), lr=lr, betas=(0.9, 0.99))

    last_loss = 0.0
    for _ in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            student_logits, _ = student(X_batch)
            with torch.no_grad():
                teacher_logits, _ = teacher(X_batch)

            hard_loss = sup_loss_fn(student_logits, y_batch)
            soft_loss = multilabel_kd_loss(
                student_logits,
                teacher_logits,
                temperature=kd_temperature
            )

            loss = kd_alpha * hard_loss + (1.0 - kd_alpha) * soft_loss

            if use_fedprox:
                loss += (mu / 2.0) * proximal_term(student, global_params, device=device)

            loss.backward()
            optimizer.step()

            last_loss = loss.item()

    return last_loss, student.state_dict()

# %% [markdown]
# ### Option 3 — Local Heterogeneous Teacher → Global Student
# 
# Teacher: persistent local client model  
# Student: shared global model copied locally each round
# 
# For now, heterogeneity is implemented as a **larger local teacher** from the same model family.

# %%
def initialize_option3_teachers(
    num_clients: int,
    table_path: str,
    teacher_filters: int = 32,
    teacher_window_size: int = 6,
    device: str = "cpu"
):
    """
    Create one persistent local teacher per client.
    """
    teachers = []
    for _ in range(num_clients):
        teacher = GenerateModel(
            table_path=table_path,
            num_of_filters=teacher_filters,
            kernel_size=teacher_window_size
        ).to(device)
        teachers.append(teacher)
    return teachers

# %%
def client_update_option3(
    global_student: nn.Module,
    local_teacher: nn.Module,
    train_loader: DataLoader,
    epochs: int = 1,
    lr: float = 0.001,
    device: str = "cpu",
    use_focal: bool = False,
    gamma: float = 2.5,
    kd_alpha: float = 0.5,
    kd_temperature: float = 2.0,
    use_fedprox: bool = False,
    mu: float = 0.01
):
    """
    Option 3:
    - local teacher persists on client
    - teacher trains on hard labels
    - student trains on supervised + KD from teacher
    - only student weights are uploaded
    - optional FedProx penalty is applied to the student only
    """
    student = copy.deepcopy(global_student).to(device)
    teacher = local_teacher.to(device)

    student.train()
    teacher.train()

    global_params = {
        k: v.detach().clone()
        for k, v in global_student.state_dict().items()
    }

    n_labels = train_loader.dataset[0][1].shape[0]
    sup_loss_fn = build_supervised_loss(
        train_loader=train_loader,
        n_labels=n_labels,
        use_focal=use_focal,
        gamma=gamma,
        device=device
    )

    student_opt = torch.optim.Adam(student.parameters(), lr=lr, betas=(0.9, 0.99))
    teacher_opt = torch.optim.Adam(teacher.parameters(), lr=lr, betas=(0.9, 0.99))

    last_loss = 0.0
    for _ in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # ---- Teacher update on hard labels ----
            teacher_opt.zero_grad()
            teacher_logits, _ = teacher(X_batch)
            teacher_loss = sup_loss_fn(teacher_logits, y_batch)
            teacher_loss.backward()
            teacher_opt.step()

            # ---- Student update on hard labels + teacher KD ----
            student_opt.zero_grad()
            student_logits, _ = student(X_batch)

            with torch.no_grad():
                teacher_logits_detached, _ = teacher(X_batch)

            hard_loss = sup_loss_fn(student_logits, y_batch)
            soft_loss = multilabel_kd_loss(
                student_logits,
                teacher_logits_detached,
                temperature=kd_temperature
            )

            loss = kd_alpha * hard_loss + (1.0 - kd_alpha) * soft_loss

            if use_fedprox:
                loss += (mu / 2.0) * proximal_term(student, global_params, device=device)

            loss.backward()
            student_opt.step()

            last_loss = loss.item()

    return last_loss, student.state_dict(), teacher

# %% [markdown]
# ### KD-Capable Federated Runner
# 
# This runner supports:
# - baseline FedAvg / FedProx
# - Option 1
# - Option 3
# 
# Server distillation is not included yet.

# %%
def run_fl_kd_once(
    split_mode: str,             # "iid" or "noniid"
    num_clients: int,
    local_epochs: int,
    config: dict,
    kd_config: dict,
    train_dataset,
    val_loader,
    test_loader,
    device,
    seed: int = 42,
    method: str = "baseline",    # "baseline", "option1", "option3"
    base_algo: str = "FedAvg",   # "FedAvg" or "FedProx"
    mu: float = 0.01,
    # non-IID params
    size_alpha: float = 0.5,
    labels_per_client: int = 10,
    bias_strength: float = 0.85
):
    start_time = time.time()

    # --- split ---
    if split_mode == "iid":
        client_datasets = split_dataset_iid(train_dataset, num_clients=num_clients, seed=seed)
    elif split_mode == "noniid":
        client_datasets = split_dataset_noniid_multilabel(
            train_dataset,
            num_clients=num_clients,
            seed=seed,
            size_alpha=size_alpha,
            labels_per_client=labels_per_client,
            bias_strength=bias_strength
        )
    else:
        raise ValueError("split_mode must be 'iid' or 'noniid'")

    client_sizes = [len(cd) for cd in client_datasets]
    client_loaders = [
        DataLoader(cd, batch_size=config["batch_size"], shuffle=True)
        for cd in client_datasets
    ]

    # --- global shared model / student ---
    global_model = GenerateModel(
        model_param_path,
        num_of_filters=config["n_filters"],
        kernel_size=config["window_size"]
    ).to(device)

    # --- persistent teachers for Option 3 ---
    local_teachers = None
    if method == "option3":
        local_teachers = initialize_option3_teachers(
            num_clients=num_clients,
            table_path=model_param_path,
            teacher_filters=kd_config["teacher_filters"],
            teacher_window_size=kd_config["teacher_window_size"],
            device=device
        )

    # --- rounds ---
    for rnd in tqdm(
        range(config["rounds"]),
        colour="blue",
        desc=f"{method} | {base_algo} | {split_mode} | K={num_clients} | E={local_epochs}"
    ):
        client_params = []

        for cid, loader in enumerate(client_loaders):
            if method == "baseline":
                client_model = copy.deepcopy(global_model)

                _, client_state, _ = client_update(
                    model=client_model,
                    train_loader=loader,
                    epochs=local_epochs,
                    lr=config["lr"],
                    device=device,
                    use_focal=config["use_focal"],
                    gamma=config["gamma"],
                    mu=mu,
                    global_params=(global_model.state_dict() if base_algo == "FedProx" else None),
                    c_global=None,
                    c_local=None,
                    algorithm=base_algo
                )
                client_params.append(client_state)

            elif method == "option1":
                _, client_state = client_update_option1(
                    global_model=global_model,
                    train_loader=loader,
                    epochs=local_epochs,
                    lr=config["lr"],
                    device=device,
                    use_focal=config["use_focal"],
                    gamma=config["gamma"],
                    kd_alpha=kd_config["kd_alpha"],
                    kd_temperature=kd_config["kd_temperature"],
                    use_fedprox=(base_algo == "FedProx"),
                    mu=mu
                )
                client_params.append(client_state)

            elif method == "option3":
                _, client_state, updated_teacher = client_update_option3(
                    global_student=global_model,
                    local_teacher=local_teachers[cid],
                    train_loader=loader,
                    epochs=local_epochs,
                    lr=config["lr"],
                    device=device,
                    use_focal=config["use_focal"],
                    gamma=config["gamma"],
                    kd_alpha=kd_config["kd_alpha"],
                    kd_temperature=kd_config["kd_temperature"],
                    use_fedprox=(base_algo == "FedProx"),
                    mu=mu
                )
                local_teachers[cid] = updated_teacher
                client_params.append(client_state)

            else:
                raise ValueError("method must be one of: baseline, option1, option3")

        # aggregate uploaded students / models
        new_params = FedAvg_weighted(client_params, client_sizes)
        global_model.load_state_dict(new_params)

    # --- eval ---
    _, per_label_thr = find_best_thresholds_per_label(global_model, val_loader, device)
    _, metrics = eval_model(global_model, device, test_loader, per_label_thr=per_label_thr)

    elapsed = time.time() - start_time
    return metrics, elapsed

# %% [markdown]
# ### Quick KD + FedProx Smoke Tests
# 
# Run a small sanity check on the current method set before launching the full sweep.
# 
# Current methods:
# - Baseline FedAvg
# - Baseline FedProx (mu = 0.01)
# - Option 1 + FedAvg
# - Option 1 + FedProx (mu = 0.01)
# - Option 3 + FedAvg
# - Option 3 + FedProx (mu = 0.01)
# 
# Setup:
# - non-IID split
# - 2 clients
# - 1 local epoch
# - fixed non-IID severity

# %%
# %%
def clean_metrics(metrics: dict):
    out = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, float)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def print_metrics(title: str, metrics: dict, time_sec: float):
    m = clean_metrics(metrics)

    print(f"\n=== {title} ===")
    print(f"Time          : {time_sec:.2f}s")
    print(f"F1 Macro      : {m.get('f1_macro', float('nan')):.4f}")
    print(f"F1 Micro      : {m.get('f1_micro', float('nan')):.4f}")
    print(f"AUC Macro     : {m.get('auc_macro', float('nan')):.4f}")
    print(f"AUC Micro     : {m.get('auc_micro', float('nan')):.4f}")
    print(f"PR-AUC Macro  : {m.get('pr_auc_macro', float('nan')):.4f}")
    print(f"PR-AUC Micro  : {m.get('pr_auc_micro', float('nan')):.4f}")


def compare_methods(results: dict):
    rows = []
    for name, (metrics, time_sec) in results.items():
        m = clean_metrics(metrics)
        rows.append({
            "method": name,
            "f1_macro": m.get("f1_macro"),
            "f1_micro": m.get("f1_micro"),
            "pr_auc_macro": m.get("pr_auc_macro"),
            "pr_auc_micro": m.get("pr_auc_micro"),
            "auc_macro": m.get("auc_macro"),
            "auc_micro": m.get("auc_micro"),
            "time_sec": round(time_sec, 2)
        })

    df = pd.DataFrame(rows).sort_values(
        by=["f1_macro", "pr_auc_macro", "auc_macro"],
        ascending=False
    ).reset_index(drop=True)

    display(df)
    return df

# %%
# Shared smoke-test settings

smoke_kwargs = dict(
    split_mode="noniid",
    num_clients=2,
    local_epochs=1,
    config=config,
    kd_config=kd_config,
    train_dataset=train_dataset,
    val_loader=val_loader,
    test_loader=test_loader,
    device=device,
    seed=42,
    mu=0.01,
    size_alpha=0.5,
    labels_per_client=10,
    bias_strength=0.85
)

# %%
# 1) Baseline FedAvg

smoke_metrics_baseline_fedavg, smoke_time_baseline_fedavg = run_fl_kd_once(
    method="baseline",
    base_algo="FedAvg",
    **smoke_kwargs
)

print_metrics("Baseline FedAvg", smoke_metrics_baseline_fedavg, smoke_time_baseline_fedavg)

# %%
# 2) Baseline FedProx

smoke_metrics_baseline_fedprox, smoke_time_baseline_fedprox = run_fl_kd_once(
    method="baseline",
    base_algo="FedProx",
    **smoke_kwargs
)

print_metrics("Baseline FedProx (mu=0.01)", smoke_metrics_baseline_fedprox, smoke_time_baseline_fedprox)

# %%
# 3) Option 1 + FedAvg

smoke_metrics_opt1_fedavg, smoke_time_opt1_fedavg = run_fl_kd_once(
    method="option1",
    base_algo="FedAvg",
    **smoke_kwargs
)

print_metrics("Option 1 + FedAvg", smoke_metrics_opt1_fedavg, smoke_time_opt1_fedavg)

# %%
# 4) Option 1 + FedProx

smoke_metrics_opt1_fedprox, smoke_time_opt1_fedprox = run_fl_kd_once(
    method="option1",
    base_algo="FedProx",
    **smoke_kwargs
)

print_metrics("Option 1 + FedProx (mu=0.01)", smoke_metrics_opt1_fedprox, smoke_time_opt1_fedprox)

# %%
# 5) Option 3 + FedAvg

smoke_metrics_opt3_fedavg, smoke_time_opt3_fedavg = run_fl_kd_once(
    method="option3",
    base_algo="FedAvg",
    **smoke_kwargs
)

print_metrics("Option 3 + FedAvg", smoke_metrics_opt3_fedavg, smoke_time_opt3_fedavg)

# %%
# 6) Option 3 + FedProx

smoke_metrics_opt3_fedprox, smoke_time_opt3_fedprox = run_fl_kd_once(
    method="option3",
    base_algo="FedProx",
    **smoke_kwargs
)

print_metrics("Option 3 + FedProx (mu=0.01)", smoke_metrics_opt3_fedprox, smoke_time_opt3_fedprox)

# %%
# Combined comparison table

smoke_df = compare_methods({
    "Baseline_FedAvg": (smoke_metrics_baseline_fedavg, smoke_time_baseline_fedavg),
    "Baseline_FedProx_mu0.01": (smoke_metrics_baseline_fedprox, smoke_time_baseline_fedprox),
    "Opt1_FedAvg": (smoke_metrics_opt1_fedavg, smoke_time_opt1_fedavg),
    "Opt1_FedProx_mu0.01": (smoke_metrics_opt1_fedprox, smoke_time_opt1_fedprox),
    "Opt3_FedAvg": (smoke_metrics_opt3_fedavg, smoke_time_opt3_fedavg),
    "Opt3_FedProx_mu0.01": (smoke_metrics_opt3_fedprox, smoke_time_opt3_fedprox),
})


