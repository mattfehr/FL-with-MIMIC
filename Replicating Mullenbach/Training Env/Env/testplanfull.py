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
import csv

# Optional: to ensure reproducibility
torch.manual_seed(42)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# %%
TESTPLAN_DIR = os.path.join("..", "History", "TestPlan")
os.makedirs(TESTPLAN_DIR, exist_ok=True)

GRID_RESULTS_CSV = os.path.join(TESTPLAN_DIR, "summary_results_fixed.csv")
GRID_RESULTS_JSON = os.path.join(TESTPLAN_DIR, "summary_results_fixed.json")

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

# %%
def save_dict_rows_to_csv(rows: list[dict], filepath: str) -> None:
    """
    Save a list of dictionaries to a CSV file.
    """
    if not rows:
        print(f"[save_dict_rows_to_csv] No rows to save for {filepath}")
        return

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(filepath, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {filepath}")

def to_serializable_row(row: dict) -> dict:
    """
    Convert metric values to plain Python scalars when possible.
    """
    cleaned = {}
    for k, v in row.items():
        if isinstance(v, (np.floating, np.integer)):
            cleaned[k] = v.item()
        elif isinstance(v, torch.Tensor):
            cleaned[k] = v.item() if v.numel() == 1 else v.detach().cpu().tolist()
        else:
            cleaned[k] = v
    return cleaned

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
    new_global = {}
    for key in global_model.keys():
        stacked = torch.stack(
            [client_dict[key].detach().cpu().float() for client_dict in client_state_dicts],
            dim=0
        )
        new_global[key] = torch.mean(stacked, dim=0)
    return new_global

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
    return FedAvg(global_model_dict, client_state_dicts)


def Scaffold(global_model_dict, client_state_dicts, c_global, c_clients_old, c_clients_new):
    """
    SCAFFOLD server update.

    Model update:
        same aggregation as FedAvg over corrected local client models

    Global control variate update:
        c <- c + average(c_i_new - c_i_old)

    Args:
        global_model_dict (dict): Current global model state_dict.
        client_state_dicts (list[dict]): Client model state_dicts after local training.
        c_global (dict): Global control variate dict, keyed by parameter name.
        c_clients_old (list[dict]): Client control variates before this round.
        c_clients_new (list[dict]): Client control variates after this round.

    Returns:
        tuple[dict, dict]:
            - new global model state_dict
            - new global control variate dict
    """
    # Global model update is just FedAvg of the corrected local models
    new_global = FedAvg(global_model_dict, client_state_dicts)

    # Global control variate update
    new_c_global = {}
    num_clients = len(c_clients_new)

    for name in c_global.keys():
        delta_c = torch.stack(
            [c_clients_new[k][name] - c_clients_old[k][name] for k in range(num_clients)],
            dim=0
        ).mean(dim=0)

        new_c_global[name] = c_global[name] + delta_c

    return new_global, new_c_global

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
    c_global: dict = None,              # for SCAFFOLD (parameter names only)
    c_local: dict = None,               # for SCAFFOLD (parameter names only)
    algorithm: str = "FedAvg",           # "FedAvg", "FedProx", or "SCAFFOLD"
    momentum: float = 0.0
) -> tuple[float, dict, dict]:
    """
    Perform local training for a single client.
    Supports FedAvg, FedProx, and SCAFFOLD.

    Returns:
        tuple:
            - final loss
            - cloned model state_dict after local training
            - updated local control variate dict (or original c_local / None)
    """
    model.to(device)
    model.train()

    n_labels = train_loader.dataset[0][1].shape[0]
    pos_weight = compute_pos_weight(train_loader, n_labels).to(device)
    if algorithm == "SCAFFOLD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

    if use_focal:
        alpha = torch.clamp(pos_weight / pos_weight.max(), min=0.1, max=0.9).to(device)
        loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    last_loss = None

    # Count optimizer steps for SCAFFOLD local control update
    step_count = 0

    for _ in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            preds, _ = model(X_batch)
            loss = loss_fn(preds, y_batch)

            # --- FedProx proximal term ---
            if algorithm == "FedProx" and global_params is not None:
                prox_term = 0.0
                for name, w in model.named_parameters():
                    w_global = global_params[name].to(device)
                    prox_term += (w - w_global).norm(2) ** 2
                loss += (mu / 2.0) * prox_term

            optimizer.zero_grad()
            loss.backward()

            # --- SCAFFOLD gradient correction ---
            if algorithm == "SCAFFOLD" and c_global is not None and c_local is not None:
                with torch.no_grad():
                    for name, w in model.named_parameters():
                        if w.grad is not None:
                            w.grad += c_global[name].to(device) - c_local[name].to(device)

            optimizer.step()
            step_count += 1
            last_loss = loss.item()

    # Safe cloned return for aggregation
    new_weights = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }

    # --- SCAFFOLD local control variate update ---
    new_c_local = c_local
    if algorithm == "SCAFFOLD" and c_global is not None and c_local is not None:
        if global_params is None:
            raise ValueError("global_params must be provided for SCAFFOLD")

        if step_count == 0:
            raise ValueError("SCAFFOLD step_count is zero; train_loader appears empty")

        new_c_local = {}
        with torch.no_grad():
            for name in c_global.keys():
                w_global = global_params[name].to(device)
                w_local = new_weights[name].to(device)
                c_g = c_global[name].to(device)
                c_l = c_local[name].to(device)

                # c_i_new = c_i_old - c + (w_global - w_local) / (K * lr)
                updated_c = c_l - c_g + (w_global - w_local) / (step_count * lr)
                new_c_local[name] = updated_c.detach().cpu().clone()

    return last_loss, new_weights, new_c_local

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
    "rounds": 100,            # communication rounds per experiment
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
# ## Full test plan loop

# %%
# Load datasets
train_dataset = TensorDataset(
    torch.load(os.path.join("..", "Data", "X_train.pt")),
    torch.load(os.path.join("..", "Data", "Y_train.pt"))
)
val_loader = load_data("val")
test_loader = load_data("test")

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
    generator = torch.Generator().manual_seed(42)
    client_datasets = random_split(train_dataset, lengths=lengths, generator=generator)
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
        c_global = {
            name: torch.zeros_like(param.detach().cpu())
            for name, param in global_model.named_parameters()
        }
        c_clients = [
            {
                name: torch.zeros_like(param.detach().cpu())
                for name, param in global_model.named_parameters()
            }
            for _ in range(num_clients)
        ]
    else:
        c_global = c_clients = None

    # --- Federated training rounds ---
    for rnd in tqdm(range(config["rounds"]), colour="blue", desc=f"{algo} | Clients={num_clients} | Epochs={local_epochs}"):
        client_params = []
        new_c_clients = []

        global_params_snapshot = {
            k: v.detach().clone()
            for k, v in global_model.state_dict().items()
        }

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
                global_params=global_params_snapshot,
                c_global=c_global if algo == "SCAFFOLD" else None,
                c_local=c_clients[idx] if algo == "SCAFFOLD" else None,
                algorithm=algo,
                momentum=config.get("momentum", 0.0)
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
            new_params, c_global = Scaffold(
                global_model.state_dict(),
                client_params,
                c_global,
                c_clients,
                new_c_clients
            )
            global_model.load_state_dict(new_params)
            c_clients = new_c_clients

    # --- Evaluate on test set using per-label thresholds from validation ---
    _, per_label_thr = find_best_thresholds_per_label(global_model, val_loader, device)
    _, metrics = eval_model(global_model, device, test_loader, per_label_thr=per_label_thr)

    elapsed = time.time() - start_time
    return metrics, elapsed


# %% [markdown]
# ### FedProx and Scaffold Tuning

# %%
# # === FedProx Hyperparameter Tuning (mu only) ===

# FEDPROX_TUNING_CSV = os.path.join(TESTPLAN_DIR, "fedprox_mu_tuning.csv")
# FEDPROX_TUNING_JSON = os.path.join(TESTPLAN_DIR, "fedprox_mu_tuning.json")

# fedprox_mus = [0.0, 0.0005, 0.001, 0.002, 0.005]

# # fixed representative setting for method-specific tuning
# tune_clients = 3
# tune_local_epochs = 2

# fedprox_tuning_results = pd.DataFrame(columns=[
#     "Function", "Tune Clients", "Tune Local Epochs", "Mu",
#     "AUC Macro", "AUC Micro",
#     "F1 Macro", "F1 Micro",
#     "PR-AUC Macro", "PR-AUC Micro",
#     "Time"
# ])

# for mu in fedprox_mus:
#     print(f"\n=== FedProx Tuning | Mu={mu} | Clients={tune_clients} | Local Epochs={tune_local_epochs} ===")

#     trial_config = config.copy()
#     trial_config["algorithm"] = "FedProx"
#     trial_config["mu"] = mu
#     trial_config["epochs"] = tune_local_epochs
#     trial_config["rounds"] = 30

#     metrics, elapsed = run_federated_experiment(
#         algo="FedProx",
#         num_clients=tune_clients,
#         local_epochs=tune_local_epochs,
#         config=trial_config,
#         train_dataset=train_dataset,
#         val_loader=val_loader,
#         test_loader=test_loader,
#         device=device
#     )

#     fedprox_tuning_results.loc[len(fedprox_tuning_results)] = [
#         "FedProx",
#         tune_clients,
#         tune_local_epochs,
#         mu,
#         metrics.get("auc_macro", None),
#         metrics.get("auc_micro", None),
#         metrics.get("f1_macro", None),
#         metrics.get("f1_micro", None),
#         metrics.get("pr_auc_macro", None),
#         metrics.get("pr_auc_micro", None),
#         elapsed
#     ]

#     fedprox_tuning_results.to_csv(FEDPROX_TUNING_CSV, index=False)
#     save_json(
#         {"rows": [to_serializable_row(r) for r in fedprox_tuning_results.to_dict(orient="records")]},
#         FEDPROX_TUNING_JSON
#     )

#     print(
#         f"Completed: Mu={mu} | "
#         f"F1_micro={metrics.get('f1_micro', float('nan')):.4f} | "
#         f"Time={elapsed:.2f}s"
#     )

# print("\n=== FedProx Tuning Complete ===")
# display(fedprox_tuning_results.sort_values("F1 Micro", ascending=False))

# %%
# # === SCAFFOLD Hyperparameter Tuning (lr only; momentum fixed at 0.0) ===

# SCAFFOLD_TUNING_CSV = os.path.join(TESTPLAN_DIR, "scaffold_hparam_tuning.csv")
# SCAFFOLD_TUNING_JSON = os.path.join(TESTPLAN_DIR, "scaffold_hparam_tuning.json")

# scaffold_lrs = [1.0, 1.25, 1.5, 2.0, 3.0]
# scaffold_momentums = [0.0]

# # fixed representative setting for method-specific tuning
# tune_clients = 3
# tune_local_epochs = 2

# scaffold_tuning_results = pd.DataFrame(columns=[
#     "Function", "Tune Clients", "Tune Local Epochs", "LR", "Momentum",
#     "AUC Macro", "AUC Micro",
#     "F1 Macro", "F1 Micro",
#     "PR-AUC Macro", "PR-AUC Micro",
#     "Time", "Status"
# ])

# for lr in scaffold_lrs:
#     for momentum in scaffold_momentums:
#         print(
#             f"\n=== SCAFFOLD Tuning | LR={lr} | Momentum={momentum} | "
#             f"Clients={tune_clients} | Local Epochs={tune_local_epochs} ==="
#         )

#         trial_config = config.copy()
#         trial_config["algorithm"] = "SCAFFOLD"
#         trial_config["lr"] = lr
#         trial_config["epochs"] = tune_local_epochs
#         trial_config["momentum"] = momentum
#         trial_config["rounds"] = 30

#         try:
#             metrics, elapsed = run_federated_experiment(
#                 algo="SCAFFOLD",
#                 num_clients=tune_clients,
#                 local_epochs=tune_local_epochs,
#                 config=trial_config,
#                 train_dataset=train_dataset,
#                 val_loader=val_loader,
#                 test_loader=test_loader,
#                 device=device
#             )

#             row = {
#                 "Function": "SCAFFOLD",
#                 "Tune Clients": tune_clients,
#                 "Tune Local Epochs": tune_local_epochs,
#                 "LR": lr,
#                 "Momentum": momentum,
#                 "AUC Macro": metrics.get("auc_macro", None),
#                 "AUC Micro": metrics.get("auc_micro", None),
#                 "F1 Macro": metrics.get("f1_macro", None),
#                 "F1 Micro": metrics.get("f1_micro", None),
#                 "PR-AUC Macro": metrics.get("pr_auc_macro", None),
#                 "PR-AUC Micro": metrics.get("pr_auc_micro", None),
#                 "Time": elapsed,
#                 "Status": "ok"
#             }

#             print(
#                 f"Completed: LR={lr} | Momentum={momentum} | "
#                 f"F1_micro={metrics.get('f1_micro', float('nan')):.4f} | "
#                 f"Time={elapsed:.2f}s"
#             )

#         except Exception as e:
#             row = {
#                 "Function": "SCAFFOLD",
#                 "Tune Clients": tune_clients,
#                 "Tune Local Epochs": tune_local_epochs,
#                 "LR": lr,
#                 "Momentum": momentum,
#                 "AUC Macro": None,
#                 "AUC Micro": None,
#                 "F1 Macro": None,
#                 "F1 Micro": None,
#                 "PR-AUC Macro": None,
#                 "PR-AUC Micro": None,
#                 "Time": None,
#                 "Status": f"failed: {type(e).__name__}: {e}"
#             }

#             print(f"FAILED: LR={lr} | Momentum={momentum} | {e}")

#         scaffold_tuning_results.loc[len(scaffold_tuning_results)] = row
#         scaffold_tuning_results.to_csv(SCAFFOLD_TUNING_CSV, index=False)
#         save_json(
#             {"rows": [to_serializable_row(r) for r in scaffold_tuning_results.to_dict(orient="records")]},
#             SCAFFOLD_TUNING_JSON
#         )

# print("\n=== SCAFFOLD Tuning Complete ===")
# display(scaffold_tuning_results.sort_values(["Status", "F1 Micro"], ascending=[True, False]))

# %% [markdown]
# ### 27 Config Run

# %%
# # === Federated Experiment Grid: FedAvg, FedProx, SCAFFOLD ===

# BEST_FEDAVG_LR = 0.002

# BEST_FEDPROX_LR = 0.002
# BEST_FEDPROX_MU = 0.001

# BEST_SCAFFOLD_LR = 1.25
# BEST_SCAFFOLD_MOMENTUM = 0.0

# results = pd.DataFrame(columns=[
#     "Function", "Clients", "Local Epochs",
#     "LR", "Mu", "Momentum",
#     "AUC Macro", "AUC Micro",
#     "F1 Macro", "F1 Micro",
#     "PR-AUC Macro", "PR-AUC Micro",
#     "Time"
# ])

# algorithms = ["FedAvg", "FedProx", "SCAFFOLD"]
# client_counts = [2, 3, 4]
# local_epochs = [1, 2, 3]

# for algo in algorithms:
#     for n_clients in client_counts:
#         for epochs in local_epochs:
#             print(f"\n=== Running {algo} | Clients={n_clients} | Local Epochs={epochs} ===")

#             trial_config = config.copy()
#             trial_config["algorithm"] = algo
#             trial_config["epochs"] = epochs

#             # method-specific tuned hyperparameters
#             if algo == "FedAvg":
#                 trial_config["lr"] = BEST_FEDAVG_LR
#                 trial_config["mu"] = None
#                 trial_config["momentum"] = None

#             elif algo == "FedProx":
#                 trial_config["lr"] = BEST_FEDPROX_LR
#                 trial_config["mu"] = BEST_FEDPROX_MU
#                 trial_config["momentum"] = None

#             elif algo == "SCAFFOLD":
#                 trial_config["lr"] = BEST_SCAFFOLD_LR
#                 trial_config["mu"] = None
#                 trial_config["momentum"] = BEST_SCAFFOLD_MOMENTUM

#             metrics, elapsed = run_federated_experiment(
#                 algo=algo,
#                 num_clients=n_clients,
#                 local_epochs=epochs,
#                 config=trial_config,
#                 train_dataset=train_dataset,
#                 val_loader=val_loader,
#                 test_loader=test_loader,
#                 device=device
#             )

#             results.loc[len(results)] = [
#                 algo,
#                 n_clients,
#                 epochs,
#                 trial_config.get("lr", None),
#                 trial_config.get("mu", None),
#                 trial_config.get("momentum", None),
#                 metrics.get("auc_macro", None),
#                 metrics.get("auc_micro", None),
#                 metrics.get("f1_macro", None),
#                 metrics.get("f1_micro", None),
#                 metrics.get("pr_auc_macro", None),
#                 metrics.get("pr_auc_micro", None),
#                 elapsed
#             ]

#             results.to_csv(GRID_RESULTS_CSV, index=False)
#             save_json(
#                 {"rows": [to_serializable_row(r) for r in results.to_dict(orient="records")]},
#                 GRID_RESULTS_JSON
#             )

#             print(
#                 f"Completed: {algo} | Clients={n_clients} | Epochs={epochs} | "
#                 f"LR={trial_config.get('lr', None)} | "
#                 f"Mu={trial_config.get('mu', None)} | "
#                 f"Momentum={trial_config.get('momentum', None)}\n"
#             )

# print("\nAll 27 configurations complete!")
# display(results)

# %% [markdown]
# ## Models for Attention Tests

# %%
# === Setup for Fixed Attention Model Training ===

output_dir = os.path.join("..", "History", "models")
os.makedirs(output_dir, exist_ok=True)

ATTN_MODELS_FIXED = {
    "central":  os.path.join(output_dir, "central_e300_best_attention_fixed.pt"),
    "fedavg":   os.path.join(output_dir, "fedavg_c2e3_best_attention_fixed.pt"),
    "fedprox":  os.path.join(output_dir, "fedprox_c3e3_mu0001_best_attention_fixed.pt"),
    "scaffold": os.path.join(output_dir, "scaffold_c4e3_lr125_m0_best_attention_fixed.pt"),
}

ATTN_CONFIGS = {
    "central": {
        "epochs": 300,
        "lr": 0.002,
    },
    "fedavg": {
        "algo": "FedAvg",
        "rounds": 100,
        "num_clients": 2,
        "local_epochs": 3,
        "lr": 0.002,
    },
    "fedprox": {
        "algo": "FedProx",
        "rounds": 100,
        "num_clients": 3,
        "local_epochs": 3,
        "lr": 0.002,
        "mu": 0.001,
    },
    "scaffold": {
        "algo": "SCAFFOLD",
        "rounds": 100,
        "num_clients": 4,
        "local_epochs": 3,
        "lr": 1.25,
        "momentum": 0.0,
    },
}

ATTN_EVAL_CSV_FIXED = os.path.join(TESTPLAN_DIR, "attention_model_test_metrics_bestconfigs_fixed.csv")
ATTN_EVAL_JSON_FIXED = os.path.join(TESTPLAN_DIR, "attention_model_test_metrics_bestconfigs_fixed.json")

print("Fixed attention model output paths:")
for k, v in ATTN_MODELS_FIXED.items():
    print(f"  {k}: {v}")

# %%
def train_centralized_for_attention_fixed(epochs=300, lr=0.002):
    train_loader = load_data("train")
    val_loader   = load_data("val")

    model = GenerateModel(
        table_path=os.path.join("..", "Model", "processed_full.w2v"),
        num_of_filters=config["n_filters"],
        kernel_size=config["window_size"]
    ).to(device)

    n_labels = train_loader.dataset[0][1].shape[0]
    pos_weight = compute_pos_weight(train_loader, n_labels).to(device)

    with torch.no_grad():
        p = pos_weight / (pos_weight + 1.0)
        prior_logit = torch.log(p / (1 - p))
        model.final.bias.copy_(prior_logit.clamp(-10, 10))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = -1.0
    best_state = None

    for ep in tqdm(range(epochs), desc="Centralized (Attention Fixed)", colour="green"):
        model.train()
        total_loss = 0.0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds, _ = model(Xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        _, per_thr = find_best_thresholds_per_label(model, val_loader, device)
        _, metrics = eval_model(model, device, val_loader, per_label_thr=per_thr)

        if metrics["f1_micro"] > best_f1:
            best_f1 = metrics["f1_micro"]
            best_state = copy.deepcopy(model.state_dict())

        print(f"Epoch {ep+1}/{epochs} | F1_micro={metrics['f1_micro']:.4f} (best={best_f1:.4f})")

    model.load_state_dict(best_state)
    return model

# %%
def train_fed_for_attention_fixed(
    algo,
    rounds=100,
    num_clients=2,
    local_epochs=3,
    lr=0.002,
    mu=0.01,
    momentum=0.0
):
    assert algo in {"FedAvg", "FedProx", "SCAFFOLD"}

    print(f"\n=== Training {algo} for Fixed Attention Tests ===")
    print(
        f"Rounds={rounds}, Clients={num_clients}, Local Epochs={local_epochs}, "
        f"LR={lr}, Mu={mu}, Momentum={momentum}"
    )

    splits = [1 / num_clients] * num_clients
    lengths = [int(len(train_dataset) * s) for s in splits[:-1]]
    lengths.append(len(train_dataset) - sum(lengths))
    generator = torch.Generator().manual_seed(42)

    client_datasets = random_split(train_dataset, lengths, generator=generator)
    client_loaders = [
        DataLoader(c, batch_size=config["batch_size"], shuffle=True)
        for c in client_datasets
    ]

    global_model = GenerateModel(
        model_param_path,
        num_of_filters=config["n_filters"],
        kernel_size=config["window_size"]
    ).to(device)

    n_labels = train_dataset[0][1].shape[0]
    full_train_loader = load_data("train")
    pos_weight = compute_pos_weight(full_train_loader, n_labels).to(device)

    with torch.no_grad():
        p = pos_weight / (pos_weight + 1.0)
        prior_logit = torch.log(p / (1 - p))
        global_model.final.bias.copy_(prior_logit.clamp(-10, 10))

    client_model = copy.deepcopy(global_model)
    val_loader = load_data("val")

    if algo == "SCAFFOLD":
        c_global = {
            name: torch.zeros_like(param.detach().cpu())
            for name, param in global_model.named_parameters()
        }
        c_clients = [
            {
                name: torch.zeros_like(param.detach().cpu())
                for name, param in global_model.named_parameters()
            }
            for _ in range(num_clients)
        ]
    else:
        c_global = c_clients = None

    best_f1 = -1.0
    best_state = None

    for rnd in tqdm(range(rounds), desc=f"{algo} FL Attention Fixed", colour="blue"):
        updates = []
        new_c_clients = []

        global_params_snapshot = {
            k: v.detach().clone()
            for k, v in global_model.state_dict().items()
        }

        for i, loader in enumerate(client_loaders):
            client_model.load_state_dict(global_model.state_dict())

            local_loss, client_state, c_local = client_update(
                model=client_model,
                train_loader=loader,
                epochs=local_epochs,
                lr=lr,
                device=device,
                use_focal=config["use_focal"],
                gamma=config["gamma"],
                mu=mu,
                global_params=global_params_snapshot,
                c_global=c_global if algo == "SCAFFOLD" else None,
                c_local=c_clients[i] if algo == "SCAFFOLD" else None,
                algorithm=algo,
                momentum=momentum
            )

            updates.append(client_state)
            new_c_clients.append(c_local)

        if algo == "FedAvg":
            global_model.load_state_dict(FedAvg(global_model.state_dict(), updates))

        elif algo == "FedProx":
            global_model.load_state_dict(FedProx(global_model.state_dict(), updates, mu=mu))

        elif algo == "SCAFFOLD":
            new_params, c_global = Scaffold(
                global_model.state_dict(),
                updates,
                c_global,
                c_clients,
                new_c_clients
            )
            global_model.load_state_dict(new_params)
            c_clients = new_c_clients

        _, per_thr = find_best_thresholds_per_label(global_model, val_loader, device)
        _, metrics = eval_model(global_model, device, val_loader, per_label_thr=per_thr)

        if metrics["f1_micro"] > best_f1:
            best_f1 = metrics["f1_micro"]
            best_state = copy.deepcopy(global_model.state_dict())

        print(f"Round {rnd+1}/{rounds} | F1_micro={metrics['f1_micro']:.4f} (best={best_f1:.4f})")

    global_model.load_state_dict(best_state)
    return global_model

# %%
# === Train Fixed Attention Models (best config per method) ===

central_model_fixed = train_centralized_for_attention_fixed(
    epochs=ATTN_CONFIGS["central"]["epochs"],
    lr=ATTN_CONFIGS["central"]["lr"]
)
torch.save(central_model_fixed.state_dict(), ATTN_MODELS_FIXED["central"])
print("Saved →", ATTN_MODELS_FIXED["central"])

fedavg_model_fixed = train_fed_for_attention_fixed(
    algo=ATTN_CONFIGS["fedavg"]["algo"],
    rounds=ATTN_CONFIGS["fedavg"]["rounds"],
    num_clients=ATTN_CONFIGS["fedavg"]["num_clients"],
    local_epochs=ATTN_CONFIGS["fedavg"]["local_epochs"],
    lr=ATTN_CONFIGS["fedavg"]["lr"]
)
torch.save(fedavg_model_fixed.state_dict(), ATTN_MODELS_FIXED["fedavg"])
print("Saved →", ATTN_MODELS_FIXED["fedavg"])

fedprox_model_fixed = train_fed_for_attention_fixed(
    algo=ATTN_CONFIGS["fedprox"]["algo"],
    rounds=ATTN_CONFIGS["fedprox"]["rounds"],
    num_clients=ATTN_CONFIGS["fedprox"]["num_clients"],
    local_epochs=ATTN_CONFIGS["fedprox"]["local_epochs"],
    lr=ATTN_CONFIGS["fedprox"]["lr"],
    mu=ATTN_CONFIGS["fedprox"]["mu"]
)
torch.save(fedprox_model_fixed.state_dict(), ATTN_MODELS_FIXED["fedprox"])
print("Saved →", ATTN_MODELS_FIXED["fedprox"])

scaffold_model_fixed = train_fed_for_attention_fixed(
    algo=ATTN_CONFIGS["scaffold"]["algo"],
    rounds=ATTN_CONFIGS["scaffold"]["rounds"],
    num_clients=ATTN_CONFIGS["scaffold"]["num_clients"],
    local_epochs=ATTN_CONFIGS["scaffold"]["local_epochs"],
    lr=ATTN_CONFIGS["scaffold"]["lr"],
    momentum=ATTN_CONFIGS["scaffold"]["momentum"]
)
torch.save(scaffold_model_fixed.state_dict(), ATTN_MODELS_FIXED["scaffold"])
print("Saved →", ATTN_MODELS_FIXED["scaffold"])

# %% [markdown]
# ### Eval Sanity Check

# %%
# === Eval helper for attention models (same logic as 27-config) ===

def eval_attention_model_on_test(model, name: str):
    """
    Evaluate a trained model on the test set using per-label thresholds
    derived from the validation set, exactly like the 27-config experiments.
    """
    # reuse loaders so we're consistent
    val_loader  = load_data("val")
    test_loader = load_data("test")

    # thresholds from validation
    _, per_label_thr = find_best_thresholds_per_label(model, val_loader, device)

    # test evaluation
    test_loss, metrics = eval_model(
        model,
        device,
        test_loader,
        per_label_thr=per_label_thr
    )

    print(f"\n📊 {name} — Test Evaluation for Attention Model")
    print(f"  Test loss      : {test_loss:.4f}")
    print(f"  AUC Macro      : {metrics['auc_macro']:.4f}")
    print(f"  AUC Micro      : {metrics['auc_micro']:.4f}")
    print(f"  F1 Macro       : {metrics['f1_macro']:.4f}")
    print(f"  F1 Micro       : {metrics['f1_micro']:.4f}")
    print(f"  PR-AUC Macro   : {metrics['pr_auc_macro']:.4f}")
    print(f"  PR-AUC Micro   : {metrics['pr_auc_micro']:.4f}")
    return metrics


# %%
# === Quick sanity check: metrics for fixed attention models ===

attn_results_fixed = pd.DataFrame(columns=[
    "Model", "AUC Macro", "AUC Micro",
    "F1 Macro", "F1 Micro",
    "PR-AUC Macro", "PR-AUC Micro"
])

for name, model in [
    ("Centralized", central_model_fixed),
    ("FedAvg",      fedavg_model_fixed),
    ("FedProx",     fedprox_model_fixed),
    ("SCAFFOLD",    scaffold_model_fixed),
]:
    metrics = eval_attention_model_on_test(model, name)
    attn_results_fixed.loc[len(attn_results_fixed)] = [
        name,
        metrics["auc_macro"],
        metrics["auc_micro"],
        metrics["f1_macro"],
        metrics["f1_micro"],
        metrics["pr_auc_macro"],
        metrics["pr_auc_micro"],
    ]

attn_results_fixed.to_csv(ATTN_EVAL_CSV_FIXED, index=False)
save_json(
    {"rows": [to_serializable_row(r) for r in attn_results_fixed.to_dict(orient="records")]},
    ATTN_EVAL_JSON_FIXED
)

print("\n=== Fixed Attention Models — Test Metrics Summary ===")
display(attn_results_fixed)


