# %% [markdown]
# # Federated vs Centralized Training Comparison
# 
# This notebook demonstrates a comparison between **Federated Learning (FL)** and **Centralized Training** for a binary classification task using PyTorch.
# 
# The routines here are condensed for clarity but preserve full functionality:
# - **Federated Training**: Clients train locally and share model weights for aggregation via `FedAvg`.
# - **Centralized Training**: A single model is trained on all combined data as a baseline.
# - **Evaluation Metrics**: F1-score, precision, recall, and loss are logged and visualized.
# 
# ---
# 

# %% [markdown]
# ## Imports

# %%
from torch.utils.data import DataLoader, TensorDataset, random_split
from gensim.models    import Word2Vec
import torch.nn.functional as F
import torch.nn as nn
import torch

import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from tqdm import tqdm
from evaluation import all_metrics

import math
import json
import os
import copy
import pandas as pd    # NEW – to store experiment results
import time             # NEW – to track runtime for each config

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

# %%
# Validation loader (used for per-label thresholding and evaluation)
val_loader = load_data(split="val")

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
    """
    Evaluate model performance on a given dataset using BCEWithLogitsLoss.
    Supports:
        - Fixed threshold (fixed_thr)
        - Tuned global threshold (tune_threshold)
        - Per-label thresholds (per_label_thr)
    Returns:
        avg_loss (float), metrics (dict) including:
        auc_macro, auc_micro, f1_macro, f1_micro,
        pr_auc_macro, pr_auc_micro, best_f1_micro, best_thr
    """
    model.eval()
    model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    all_pred_raw = torch.empty(0, dtype=torch.float32, device=device)
    all_labels = torch.empty(0, dtype=torch.float32, device=device)
    total_loss = 0.0

    # ---- Forward pass ----
    for X_batch, y_batch in data_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds, _ = model(X_batch)
        loss = loss_fn(preds, y_batch)
        total_loss += loss.item()

        all_pred_raw = torch.cat([all_pred_raw, preds], dim=0)
        all_labels = torch.cat([all_labels, y_batch], dim=0)

    avg_loss = total_loss / len(data_loader)

    # ---- Diagnostics ----
    logit_min, logit_max = all_pred_raw.min().item(), all_pred_raw.max().item()
    sigmoid_vals = torch.sigmoid(all_pred_raw)
    print(f"[Eval Debug] Logit range: {logit_min:.2f} to {logit_max:.2f} | "
          f"Sigmoid mean={sigmoid_vals.mean():.3f}, std={sigmoid_vals.std():.3f}")

    # ---- Apply thresholds ----
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

    # ---- Compute metrics if per_label_thr used ----
    if per_label_thr is not None:
        metrics = all_metrics(
            yhat=pred_labels.cpu().numpy(),
            y=all_labels.cpu().numpy(),
            yhat_raw=all_pred_raw.cpu().numpy()
        )

    # ---- Add PR-AUC and F1 info ----
    pr_macro = metrics.get("pr_auc_macro", None)
    pr_micro = metrics.get("pr_auc_micro", None)
    auc_macro = metrics.get("auc_macro", None)
    auc_micro = metrics.get("auc_micro", None)

    avg_pred_labels = pred_labels.sum(dim=1).float().mean().item()

    print(
        f"[Eval] Avg loss={avg_loss:.4f} | "
        f"F1_micro={metrics['f1_micro']:.4f} | F1_macro={metrics['f1_macro']:.4f} | "
        f"AUC_macro={auc_macro:.4f} | AUC_micro={auc_micro:.4f} | "
        f"PR-AUC_macro={pr_macro:.4f} | PR-AUC_micro={pr_micro:.4f} | "
        f"Best_F1={best_f1 if best_f1 else metrics['f1_micro']:.4f} @ thr={best_thr} | "
        f"Avg labels/sample={avg_pred_labels:.2f}"
    )

    # ---- Store for history ----
    metrics["best_f1_micro"] = best_f1 if best_f1 else metrics["f1_micro"]
    metrics["best_thr"] = best_thr

    return avg_loss, metrics


# %% [markdown]
# ## Full test plan loop

# %%
def run_federated_experiment(algo, num_clients, local_epochs, config, train_dataset, val_loader, test_loader, device):
    """
    Runs one federated configuration (FedAvg, FedProx, or SCAFFOLD)
    and returns evaluation metrics on the test set.
    """
    start_time = time.time()

    # --- Split dataset into clients dynamically ---
    splits = [1 / num_clients] * num_clients
    lengths = [int(len(train_dataset) * s) for s in splits[:-1]]
    lengths.append(len(train_dataset) - sum(lengths))
    client_datasets = random_split(train_dataset, lengths=lengths)
    c_loaders = [DataLoader(c, batch_size=config["batch_size"], shuffle=True) for c in client_datasets]

    # --- Initialize global and client models ---
    global_model = GenerateModel(
        model_param_path,
        num_of_filters=config["n_filters"],
        kernel_size=config["window_size"]
    ).to(device)

    client_model = copy.deepcopy(global_model)

    # --- Initialize control variates if SCAFFOLD ---
    if algo == "SCAFFOLD":
        c_global = {k: torch.zeros_like(v) for k, v in global_model.state_dict().items()}
        c_clients = [{k: torch.zeros_like(v) for k, v in global_model.state_dict().items()} for _ in range(num_clients)]
    else:
        c_global = c_clients = None

    # --- Federated training rounds ---
    for rnd in tqdm(range(config["rounds"]), colour="blue", desc=f"{algo} | Clients={num_clients} | Epochs={local_epochs}"):
        client_params = []
        new_c_clients = []

        # ---- Each client trains locally ----
        for idx, loader in enumerate(c_loaders):
            client_model.load_state_dict(global_model.state_dict())

            local_loss, client_state, c_local = client_update(
                model=client_model,
                train_loader=loader,
                epochs=local_epochs,
                lr=config["lr"],
                device=device,
                use_focal=config["use_focal"],
                gamma=config["gamma"],
                mu=config["mu"],
                global_params=global_model.state_dict(),
                c_global=c_global if algo == "SCAFFOLD" else None,
                c_local=c_clients[idx] if algo == "SCAFFOLD" else None,
                algorithm=algo
            )

            client_params.append(client_state)
            new_c_clients.append(c_local)

        # ---- Aggregate updates ----
        if algo == "FedAvg":
            new_params = FedAvg(global_model.state_dict(), client_params)
            global_model.load_state_dict(new_params)

        elif algo == "FedProx":
            new_params = FedProx(global_model.state_dict(), client_params, mu=config["mu"])
            global_model.load_state_dict(new_params)

        elif algo == "SCAFFOLD":
            new_params, c_global, c_clients = Scaffold(
                global_model.state_dict(),
                client_params,
                c_global,
                c_clients,
                lr=config["lr"],
                num_clients=num_clients
            )
            global_model.load_state_dict(new_params)

    # --- Evaluate on test set using per-label thresholds from validation ---
    _, per_label_thr = find_best_thresholds_per_label(global_model, val_loader, device)
    _, metrics = eval_model(global_model, device, test_loader, per_label_thr=per_label_thr)

    elapsed = time.time() - start_time
    return metrics, elapsed


# %%
# === Federated Experiment Grid: FedAvg, FedProx, SCAFFOLD ===

results = pd.DataFrame(columns=[
    "Function", "Clients", "Local Epochs",
    "AUC Macro", "AUC Micro",
    "F1 Macro", "F1 Micro",
    "PR-AUC Macro", "PR-AUC Micro",
    "Time"
])

# Load datasets
train_dataset = TensorDataset(
    torch.load(os.path.join("..", "Data", "X_train.pt")),
    torch.load(os.path.join("..", "Data", "Y_train.pt"))
)
val_loader = load_data("val")
test_loader = load_data("test")

algorithms = ["FedAvg", "FedProx", "SCAFFOLD"]
client_counts = [2, 3, 4]
local_epochs = [1, 2, 3]

for algo in algorithms:
    for n_clients in client_counts:
        for epochs in local_epochs:
            print(f"\n=== Running {algo} | Clients={n_clients} | Local Epochs={epochs} ===")
            config["algorithm"] = algo
            config["epochs"] = epochs

            metrics, elapsed = run_federated_experiment(
                algo=algo,
                num_clients=n_clients,
                local_epochs=epochs,
                config=config.copy(),
                train_dataset=train_dataset,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device
            )

            results.loc[len(results)] = [
                algo, n_clients, epochs,
                metrics.get("auc_macro", None),
                metrics.get("auc_micro", None),
                metrics.get("f1_macro", None),
                metrics.get("f1_micro", None),
                metrics.get("pr_auc_macro", None),
                metrics.get("pr_auc_micro", None),
                elapsed
            ]

            # Save after each run to preserve progress
            results.to_csv("../History/summary_results.csv", index=False)
            print(f"✅ Completed: {algo} | Clients={n_clients} | Epochs={epochs}\n")

print("\n🎯 All 27 configurations complete!")
print(results)



