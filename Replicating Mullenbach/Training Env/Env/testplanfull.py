# %% [markdown]
# # Test Plan

# %% [markdown]
# ## Imports and Set Ups
# - Imports core libraries (PyTorch, NumPy, etc.)
# - Defines helper functions (`save_json`, `load_json`, `set_seed`)
# - Adds a simple runtime tracker

# %%
# Device set up and imports

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from gensim.models import Word2Vec

import numpy as np
import random
import json
import os
import time
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import copy
import pandas as pd

# Custom metrics from existing evaluation.py
from evaluation import all_metrics


# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## Utils

# %%
# Helper Functions

def set_seed(seed: int = 42):
    """
    Set all random seeds for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(data: dict, filepath: str):
    """
    Save a Python dictionary as a JSON file.
    Creates parent directories automatically.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> dict:
    """
    Load a JSON file into a Python dictionary.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


class Timer:
    """
    Simple context manager for timing code blocks.
    Example:
        with Timer() as t:
            run_training()
        print(t.elapsed)
    """
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.start

#check
set_seed(42)

# %%
def compute_pos_weight(train_loader, n_labels):
    pos = torch.zeros(n_labels)
    total = 0
    for _, y in train_loader:
        pos += y.sum(dim=0)
        total += y.shape[0]
    neg = total - pos
    return (neg / pos.clamp_min(1.0)).float()

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
    per_label_thr=None,
    fixed_thr: float = 0.3,
    tune_threshold: bool = False
):
    """
    Evaluate a model using BCEWithLogitsLoss and report:
        - Macro-F1, Micro-F1
        - AUROC (macro/micro)
        - PR-AUC (macro/micro)
    
    Args:
        model: trained model
        device: 'cpu' or 'cuda'
        data_loader: DataLoader for validation/test
        per_label_thr: Tensor of per-label thresholds (optional)
        fixed_thr: default threshold if not tuned
        tune_threshold: if True, sweeps thresholds to find best global one
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
    sigmoid_vals = torch.sigmoid(all_pred_raw)

    # ---- Thresholding ----
    if per_label_thr is not None:
        pred_labels = (sigmoid_vals >= per_label_thr.to(device)).long()
    else:
        thr = fixed_thr
        if tune_threshold:
            _, thr = find_best_threshold(model, data_loader, device)
        pred_labels = (sigmoid_vals >= thr).long()

    # ---- Compute metrics ----
    metrics = all_metrics(
        yhat=pred_labels.cpu().numpy(),
        y=all_labels.cpu().numpy(),
        yhat_raw=all_pred_raw.cpu().numpy()
    )

    # Expected keys in metrics: f1_macro, f1_micro, auc_macro, auc_micro
    # Add PR-AUC if missing
    from sklearn.metrics import average_precision_score

    if "prauc_macro" not in metrics or "prauc_micro" not in metrics:
        y_true = all_labels.cpu().numpy()
        y_score = sigmoid_vals.cpu().numpy()
        try:
            metrics["prauc_macro"] = average_precision_score(y_true, y_score, average="macro")
            metrics["prauc_micro"] = average_precision_score(y_true, y_score, average="micro")
        except ValueError:
            metrics["prauc_macro"], metrics["prauc_micro"] = 0.0, 0.0

    # ---- Summary output ----
    print(
        f"[Eval] Loss={avg_loss:.4f} | "
        f"F1_micro={metrics.get('f1_micro',0):.4f} | "
        f"F1_macro={metrics.get('f1_macro',0):.4f} | "
        f"AUC_micro={metrics.get('auc_micro',0):.4f} | "
        f"AUC_macro={metrics.get('auc_macro',0):.4f} | "
        f"PR_micro={metrics.get('prauc_micro',0):.4f} | "
        f"PR_macro={metrics.get('prauc_macro',0):.4f}"
    )

    metrics["avg_loss"] = avg_loss
    return avg_loss, metrics


# %% [markdown]
# ## Data Loading and Dirichlet Partitioning
# 
# Implements α-controlled non-IID data partitioning using a Dirichlet distribution.
# - α = 0.2 → highly non-IID (clients see different label distributions)
# - α = 1.0 → more IID (clients have similar label distributions)
# 
# Partitions and DataLoaders are saved to disk under `/History/TestPlan/...` for reproducibility.

# %%
def load_full_dataset():
    """
    Load preprocessed train tensors (X, Y).
    Returns:
        X_train, Y_train: torch.Tensor pairs
    """
    X_train = torch.load(os.path.join("..", "Data", "X_train.pt"))
    Y_train = torch.load(os.path.join("..", "Data", "Y_train.pt"))
    return X_train, Y_train


def dirichlet_partition(Y: torch.Tensor, alpha: float, num_clients: int):
    """
    Partition dataset indices across clients using Dirichlet sampling over labels.

    Args:
        Y (torch.Tensor): Label tensor of shape (N, num_labels)
        alpha (float): Dirichlet concentration parameter controlling non-IIDness
        num_clients (int): Number of simulated clients

    Returns:
        list[list[int]]: List of index lists per client
    """
    Y_np = Y.cpu().numpy()
    num_labels = Y_np.shape[1]

    # Draw label distributions for each label across clients
    label_distrib = np.random.dirichlet([alpha] * num_clients, num_labels)

    client_indices = [[] for _ in range(num_clients)]
    for i, labels in enumerate(Y_np):
        for j in np.where(labels > 0)[0]:
            probs = label_distrib[j]
            client_idx = np.random.choice(num_clients, p=probs)
            client_indices[client_idx].append(i)

    # Remove duplicates and shuffle
    for idx_list in client_indices:
        np.random.shuffle(idx_list)
        # Optional deduplication to avoid overlaps (rare)
        idx_set = list(dict.fromkeys(idx_list))
        idx_list[:] = idx_set

    return client_indices


def create_client_loaders(X, Y, alpha, num_clients, batch_size, save_path=None):
    """
    Create DataLoaders per client according to a Dirichlet partition.
    Saves partition indices to disk if save_path is provided.
    """
    client_indices = dirichlet_partition(Y, alpha, num_clients)
    client_loaders = []

    for cid, indices in enumerate(client_indices):
        subset = TensorDataset(X[indices], Y[indices])
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)

    # Save partition file for reproducibility
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(client_indices, save_path)
        print(f"Saved partition indices → {save_path}")

    return client_loaders

# %%
# Example parameters — these will later come from experiment config
alpha_example = 0.2
num_clients_example = 3
batch_size_example = 32

X_train, Y_train = load_full_dataset()

partition_path = os.path.join("..", "History", "TestPlan", "FedAvg", f"alpha_{alpha_example}", "data_partition.pt")

client_loaders = create_client_loaders(
    X_train, Y_train,
    alpha=alpha_example,
    num_clients=num_clients_example,
    batch_size=batch_size_example,
    save_path=partition_path
)

# Inspect distribution
for i, c_loader in enumerate(client_loaders):
    print(f"Client {i+1}: {len(c_loader.dataset)} samples")

# %% [markdown]
# ### Val and Test Loaders
# 
# - Loads the fixed validation and test splits.
# - These remain the same for all algorithms (FedAvg, FedProx, SCAFFOLD, etc.)
# - and are not part of the Dirichlet partitioning

# %%
def load_eval_datasets(batch_size: int = 32):
    """
    Load validation and test DataLoaders.
    Args:
        batch_size (int): Batch size for evaluation loaders.
    Returns:
        (val_loader, test_loader)
    """
    X_val = torch.load(os.path.join("..", "Data", "X_val.pt"))
    Y_val = torch.load(os.path.join("..", "Data", "Y_val.pt"))
    X_test = torch.load(os.path.join("..", "Data", "X_test.pt"))
    Y_test = torch.load(os.path.join("..", "Data", "Y_test.pt"))

    val_loader = DataLoader(
        TensorDataset(X_val, Y_val),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    test_loader = DataLoader(
        TensorDataset(X_test, Y_test),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    print(f"Validation set: {len(val_loader.dataset)} samples")
    print(f"Test set:       {len(test_loader.dataset)} samples")

    return val_loader, test_loader


# Example usage
val_loader, test_loader = load_eval_datasets(batch_size=batch_size_example)

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


# Example: Instantiate the model
model = GenerateModel(table_path=os.path.join("..", "Model", "processed_full.w2v")).to(device)
model

# %% [markdown]
# ### FL Algorithms
# 
# This section defines the FL aggregation strategies used in the MIMIC Test Plan:
# - **FedAvg** — baseline method (averages client model parameters)
# - **FedProx** — adds a proximal term to handle non-IID data
# - **SCAFFOLD** — introduces control variates (c_global, c_local) to correct client drift

# %%
# === FedAvg (baseline) ===
def FedAvg(global_model: dict, client_state_dicts: list[dict]) -> dict:
    """
    Perform Federated Averaging (FedAvg) on parameter dictionaries.
    """
    for key in global_model.keys():
        stacked = torch.stack(
            [client_dict[key].float() for client_dict in client_state_dicts],
            dim=0
        )
        global_model[key] = torch.mean(stacked, dim=0)
    return global_model

# %%
# === FedProx (adds proximal term) ===
def FedProx_update(model, global_params, mu=0.01):
    """
    Compute the FedProx proximal term.
    Args:
        model: Local model (nn.Module)
        global_params: Global model parameters (Iterable[Tensor])
        mu: Regularization strength (float)
    Returns:
        torch.Tensor: Proximal regularization loss term
    """
    prox_loss = 0.0
    for p, g in zip(model.parameters(), global_params):
        prox_loss += ((p - g.detach()) ** 2).sum()
    return mu / 2 * prox_loss

# %%
# === SCAFFOLD (with control variates) ===
class ScaffoldController:
    """
    Implements SCAFFOLD correction using control variates.
    Tracks c_global and each client's c_local.
    """
    def __init__(self, model):
        # initialize control variates for global and each client
        self.c_global = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
        self.c_local = {}  # {client_id: {param_name: tensor}}

    def init_client(self, client_id, model):
        """Initialize local control variates for a client."""
        self.c_local[client_id] = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}

    def apply_update(self, client_id, model, global_model, lr):
        """
        Apply SCAFFOLD correction to model gradients before local update.
        """
        for (name, param), (name_g, param_g) in zip(model.named_parameters(), global_model.named_parameters()):
            if param.grad is not None:
                param.grad += self.c_global[name_g] - self.c_local[client_id][name]


    def update_controls(self, client_id, old_state, new_state, lr, local_epochs):
        """
        Update both local and global control variates after client training.
        """
        c_local_old = self.c_local[client_id]
        delta_c = {}
        for key in new_state.keys():
            delta_w = new_state[key] - old_state[key]
            delta_c[key] = -(1.0 / (local_epochs * lr)) * delta_w
            c_local_old[key] = c_local_old[key] + delta_c[key] - self.c_global[key]

        # global update = mean of all local updates (to be done on server side)
        return delta_c
    
# Example instantiation (used during training setup)
# scaffold_ctrl = ScaffoldController(Global_Model)
# scaffold_ctrl.init_client(0, Global_Model)

# %%
def client_update(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 1,
    lr: float = 0.001,
    device: str = "cpu",
    # --- FL algorithm options ---
    global_model: nn.Module = None,
    use_fedprox: bool = False,
    mu: float = 0.01,
    scaffold_ctrl=None,
    client_id: int = None
) -> tuple[float, dict]:
    """
    Perform local training for a single client using BCEWithLogitsLoss.
    Supports:
        - FedAvg (default)
        - FedProx (proximal regularization)
        - SCAFFOLD (control variates correction)
    """
    model.to(device)
    model.train()

    # ---- Compute pos_weight for class imbalance ----
    n_labels = train_loader.dataset[0][1].shape[0]
    pos_weight = compute_pos_weight(train_loader, n_labels).to(device)

    # ---- Loss and optimizer ----
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

    # ---- Store initial weights if using SCAFFOLD ----
    old_state = copy.deepcopy(model.state_dict()) if scaffold_ctrl else None

    # ---- Local training ----
    for _ in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            preds, _ = model(X_batch)
            loss = loss_fn(preds, y_batch)

            # --- FedProx proximal term ---
            if use_fedprox and global_model is not None:
                loss += FedProx_update(model, global_model.parameters(), mu)

            optimizer.zero_grad()
            loss.backward()

            # --- SCAFFOLD gradient correction ---
            if scaffold_ctrl is not None and client_id is not None:
                scaffold_ctrl.apply_update(client_id, model, global_model, lr)

            optimizer.step()

    # ---- Update SCAFFOLD control variates ----
    if scaffold_ctrl is not None and client_id is not None:
        new_state = copy.deepcopy(model.state_dict())
        scaffold_ctrl.update_controls(client_id, old_state, new_state, lr, epochs)

    return loss.item(), model.state_dict()


# %% [markdown]
# ## FL Training

# %% [markdown]
# ### Set up

# %%
# Current FL run configuration

config = {
    # --- Data & Clients ---
    "alpha": 0.2,               # Dirichlet concentration parameter (0.2 = highly non-IID, 1.0 = more IID)
    "num_clients": 3,           # total number of clients simulated
    "clients_per_round": 2,     # ~50% of clients per communication round

    # --- Training Parameters ---
    "local_epochs": 1,          # how many local epochs each client trains per round
    "rounds": 100,              # total global communication rounds
    "batch_size": 32,           # batch size per client
    "lr": 2e-5,                 # learning rate
    "max_seq_len": 256,         # sequence length (ensure matching dataset)
    
    # --- Model Parameters ---
    "n_filters": 21,            # convolutional filters in ConvAttnPool
    "window_size": 6,           # kernel size for Conv1D

    # --- Logging / Reproducibility ---
    "seed": 42,                 # random seed for reproducibility
    "save_dir": "../History/TestPlan/FedAvg",  # root directory for saving logs and checkpoints

    # --- Misc ---
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Model embedding table
model_param_path = os.path.join("..", "Model", "processed_full.w2v")

print(f"Configuration loaded. Using device: {config['device']}")


# %%
# Dirichlet Partitioning

# Load preprocessed training tensors
X_train = torch.load(os.path.join("..", "Data", "X_train.pt"))
Y_train = torch.load(os.path.join("..", "Data", "Y_train.pt"))

# Unpack config parameters
alpha = config["alpha"]
num_clients = config["num_clients"]
batch_size = config["batch_size"]

# Save path for reproducibility
partition_path = os.path.join(
    config["save_dir"],
    f"alpha_{alpha}",
    "data_partition.pt"
)

# Create client data loaders using Dirichlet partitioning
client_loaders = create_client_loaders(
    X=X_train,
    Y=Y_train,
    alpha=alpha,
    num_clients=num_clients,
    batch_size=batch_size,
    save_path=partition_path
)

print(f"Created {num_clients} client loaders (Dirichlet α={alpha})")

# Validation and Test loaders (fixed for all experiments)
val_loader, test_loader = load_eval_datasets(batch_size=batch_size)

# Inspect data distribution per client
for i, loader in enumerate(client_loaders):
    print(f"Client {i+1}: {len(loader.dataset)} samples")

# %%
# Optional: visualize label distribution per client
from collections import Counter

for idx, loader in enumerate(client_loaders):
    all_codes = [
        code
        for _, y in loader.dataset
        for code in torch.where(y > 0)[0].tolist()
    ]
    counts = Counter(all_codes)
    labels, values = zip(*counts.items())

    plt.figure(figsize=(15, 5))
    plt.title(f"Client {idx + 1} Label Distribution (α={alpha})")
    plt.bar(range(len(labels)), values, 1)
    plt.xticks(range(len(labels)), labels)
    plt.show()

# %%
# Model Initialization

device = config["device"]
print(f"Using device: {device}")

# Create output directory for this experiment
alpha_dir = os.path.join(config["save_dir"], f"alpha_{config['alpha']}")
os.makedirs(alpha_dir, exist_ok=True)

# Initialize Global and Client models
Global_Model = GenerateModel(
    table_path=model_param_path,
    num_of_filters=config["n_filters"],
    kernel_size=config["window_size"]
).to(device)

Client_Model = GenerateModel(
    table_path=model_param_path,
    num_of_filters=config["n_filters"],
    kernel_size=config["window_size"]
).to(device)

# Compute class imbalance weights using one client's data
n_labels = val_loader.dataset[0][1].shape[0]
pos_weight = compute_pos_weight(client_loaders[0], n_labels).to(device)

# ---- Bias initialization (helps stabilize early training) ----
with torch.no_grad():
    p = pos_weight / (pos_weight + 1.0)
    prior_logit = torch.log(p / (1 - p))
    Global_Model.final.bias.copy_(prior_logit.clamp(-10, 10))
    Client_Model.final.bias.copy_(prior_logit.clamp(-10, 10))

print("Models initialized and bias corrected.")
print(f"Global and client models ready for α={config['alpha']} at {alpha_dir}")


# %% [markdown]
# ### Main FL Training Loop

# %%
# Initialize history tracking
history = {
    "local_loss": {cid: [] for cid in range(config["num_clients"])},
    "global_loss": [],
    "global_metrics": []
}

max_auc_macro = 0.0
max_f1_micro = 0.0

# %%
# FL Training Loop

# Save directory for this alpha
alpha_dir = os.path.join(config["save_dir"], f"alpha_{config['alpha']}")
os.makedirs(alpha_dir, exist_ok=True)

# Evaluation interval
target_evals = 100
eval_interval = max(1, round(config["rounds"] / target_evals))

print(f"Starting Federated Training (FedAvg) | α={config['alpha']} | Clients={config['num_clients']}")

for rnd in tqdm(range(config["rounds"]), desc="Federated Rounds", colour="blue"):
    client_params = []
    total_local_loss = 0.0

    # ---- Randomly select subset of clients (~50%) ----
    selected_clients = random.sample(
        range(config["num_clients"]),
        k=config["clients_per_round"]
    )

    # ---- Local Training ----
    for cid in selected_clients:
        Client_Model.load_state_dict(Global_Model.state_dict())

        local_loss, c_params = client_update(
            model=Client_Model,
            train_loader=client_loaders[cid],
            epochs=config["local_epochs"],
            lr=config["lr"],
            device=config["device"]
        )

        client_params.append(c_params)
        history["local_loss"][cid].append(local_loss)
        total_local_loss += local_loss

    avg_local_loss = total_local_loss / len(selected_clients)

    # ---- Aggregation step (FedAvg) ----
    new_params = FedAvg(Global_Model.state_dict(), client_params)
    Global_Model.load_state_dict(new_params)

    # ---- Periodic Evaluation ----
    if (rnd + 1) % eval_interval == 0 or rnd == config["rounds"] - 1:
        _, per_label_thr = find_best_thresholds_per_label(Global_Model, val_loader, config["device"])
        g_loss, metrics = eval_model(Global_Model, config["device"], val_loader, per_label_thr=per_label_thr)

        history["global_loss"].append(g_loss)
        history["global_metrics"].append(metrics)

        # ---- Save progress ----
        save_json(history, os.path.join(alpha_dir, "metrics.json"))
        torch.save(Global_Model.state_dict(), os.path.join(alpha_dir, "latest_model.pt"))

        # Track best performing models
        if metrics["auc_macro"] > max_auc_macro:
            max_auc_macro = metrics["auc_macro"]
            torch.save(Global_Model.state_dict(), os.path.join(alpha_dir, "max_auc.pt"))

        if metrics["f1_micro"] > max_f1_micro:
            max_f1_micro = metrics["f1_micro"]
            torch.save(Global_Model.state_dict(), os.path.join(alpha_dir, "max_f1_micro.pt"))

        print(
            f"Round {rnd+1}/{config['rounds']} | "
            f"Clients={selected_clients} | "
            f"Avg Local Loss={avg_local_loss:.4f} | "
            f"Val F1_micro={metrics['f1_micro']:.4f} | "
            f"AUC_macro={metrics['auc_macro']:.4f}"
        )

print("Federated training complete!")

# ---- Final best threshold check ----
best_f1, best_thr = find_best_threshold(Global_Model, val_loader, config["device"])
print(f"Best validation threshold: {best_thr} | F1_micro={best_f1:.4f}")


# %%
# Final Eval on Test Set

print("\n=== Evaluating Final Global Model on Test Set ===")

# Compute per-label thresholds from validation set for fair evaluation
_, per_label_thr = find_best_thresholds_per_label(Global_Model, val_loader, config["device"])

# Evaluate on test data
test_loss, test_metrics = eval_model(
    Global_Model,
    config["device"],
    test_loader,
    per_label_thr=per_label_thr
)

# Display test results
print(
    f"Test Results — "
    f"F1_micro={test_metrics['f1_micro']:.4f}, "
    f"F1_macro={test_metrics['f1_macro']:.4f}, "
    f"AUC_micro={test_metrics['auc_micro']:.4f}, "
    f"AUC_macro={test_metrics['auc_macro']:.4f}"
)

# ---- Prepare result entry ----
result_entry = {
    "alpha": config["alpha"],
    "num_clients": config["num_clients"],
    "clients_per_round": config["clients_per_round"],
    "local_epochs": config["local_epochs"],
    "batch_size": config["batch_size"],
    "lr": config["lr"],
    "seed": config["seed"],
    "F1_micro": round(test_metrics["f1_micro"], 4),
    "F1_macro": round(test_metrics["f1_macro"], 4),
    "AUC_micro": round(test_metrics["auc_micro"], 4),
    "AUC_macro": round(test_metrics["auc_macro"], 4),
    "PR_micro": round(test_metrics.get("prauc_micro", 0.0), 4),
    "PR_macro": round(test_metrics.get("prauc_macro", 0.0), 4),
    "test_loss": round(test_loss, 4)
}

# ---- Append to FedAvg summary CSV ----

summary_path = os.path.join(config["save_dir"], "summary.csv")

if os.path.exists(summary_path):
    df = pd.read_csv(summary_path)
    df = pd.concat([df, pd.DataFrame([result_entry])], ignore_index=True)
else:
    df = pd.DataFrame([result_entry])

df.to_csv(summary_path, index=False)
print(f"Results saved to {summary_path}")

# ---- Also append to global summary across algorithms ----
global_summary_path = os.path.join(os.path.dirname(config["save_dir"]), "global_summary.csv")

if os.path.exists(global_summary_path):
    df_global = pd.read_csv(global_summary_path)
    df_global = pd.concat([df_global, pd.DataFrame([result_entry | {"algorithm": "FedAvg"}])], ignore_index=True)
else:
    df_global = pd.DataFrame([result_entry | {"algorithm": "FedAvg"}])

df_global.to_csv(global_summary_path, index=False)
print(f"Global summary updated at {global_summary_path}")


# %% [markdown]
# ## Wrapper Function for loop

# %%
def run_fl_experiment(config: dict):
    """
    Run a full FL experiment for one algorithm and configuration.
    Automatically handles:
        - Dirichlet data partitioning
        - Model initialization
        - Training with FedAvg / FedProx / SCAFFOLD
        - Test evaluation
        - Logging + summary saving
    """

    # === 1. Set Seed and Prepare ===
    set_seed(config["seed"])
    algo_name = config["algorithm"]
    print(f"\nStarting {algo_name} | α={config['alpha']} | seed={config['seed']} | device={config['device']}")

    # === 2. Data Partitioning ===
    X_train = torch.load(os.path.join("..", "Data", "X_train.pt"))
    Y_train = torch.load(os.path.join("..", "Data", "Y_train.pt"))
    algo_dir = os.path.join(config["save_root"], algo_name)
    alpha_dir = os.path.join(algo_dir, f"alpha_{config['alpha']}")
    os.makedirs(alpha_dir, exist_ok=True)

    partition_path = os.path.join(alpha_dir, f"data_partition_seed_{config['seed']}.pt")

    client_loaders = create_client_loaders(
        X=X_train,
        Y=Y_train,
        alpha=config["alpha"],
        num_clients=config["num_clients"],
        batch_size=config["batch_size"],
        save_path=partition_path
    )

    val_loader, test_loader = load_eval_datasets(batch_size=config["batch_size"])

    # === 3. Model Initialization ===
    model_param_path = os.path.join("..", "Model", "processed_full.w2v")

    Global_Model = GenerateModel(
        table_path=model_param_path,
        num_of_filters=config["n_filters"],
        kernel_size=config["window_size"]
    ).to(config["device"])

    Client_Model = GenerateModel(
        table_path=model_param_path,
        num_of_filters=config["n_filters"],
        kernel_size=config["window_size"]
    ).to(config["device"])

    n_labels = val_loader.dataset[0][1].shape[0]
    pos_weight = compute_pos_weight(client_loaders[0], n_labels).to(config["device"])

    with torch.no_grad():
        p = pos_weight / (pos_weight + 1.0)
        prior_logit = torch.log(p / (1 - p))
        Global_Model.final.bias.copy_(prior_logit.clamp(-10, 10))
        Client_Model.final.bias.copy_(prior_logit.clamp(-10, 10))

    # === 4. Initialize optional SCAFFOLD controller ===
    scaffold_ctrl = None
    if algo_name.lower() == "scaffold":
        scaffold_ctrl = ScaffoldController(Global_Model)
        for cid in range(config["num_clients"]):
            scaffold_ctrl.init_client(cid, Global_Model)

    # === 5. History and metrics ===
    history = {"local_loss": {cid: [] for cid in range(config["num_clients"])},
               "global_loss": [], "global_metrics": []}
    max_auc_macro, max_f1_micro = 0.0, 0.0
    eval_interval = max(1, round(config["rounds"] / 100))

    # === 6. Training Loop ===
    for rnd in tqdm(range(config["rounds"]), desc=f"{algo_name} α={config['alpha']} seed={config['seed']}", colour="blue"):
        client_params, total_local_loss = [], 0.0
        selected_clients = random.sample(range(config["num_clients"]), k=config["clients_per_round"])

        for cid in selected_clients:
            Client_Model.load_state_dict(Global_Model.state_dict())
            local_loss, c_params = client_update(
                model=Client_Model,
                train_loader=client_loaders[cid],
                epochs=config["local_epochs"],
                lr=config["lr"],
                device=config["device"],
                global_model=Global_Model if algo_name.lower() in ["fedprox", "scaffold"] else None,
                use_fedprox=(algo_name.lower() == "fedprox"),
                mu=config.get("mu", 0.01),
                scaffold_ctrl=scaffold_ctrl if algo_name.lower() == "scaffold" else None,
                client_id=cid if algo_name.lower() == "scaffold" else None
            )
            client_params.append(c_params)
            history["local_loss"][cid].append(local_loss)
            total_local_loss += local_loss

        avg_local_loss = total_local_loss / len(selected_clients)
        new_params = FedAvg(Global_Model.state_dict(), client_params)
        Global_Model.load_state_dict(new_params)

        if algo_name.lower() == "scaffold":
            delta_cs = [scaffold_ctrl.c_local[cid] for cid in selected_clients]
            for key in scaffold_ctrl.c_global.keys():
                scaffold_ctrl.c_global[key] = torch.mean(torch.stack([d[key] for d in delta_cs]), dim=0)

        if (rnd + 1) % eval_interval == 0 or rnd == config["rounds"] - 1:
            _, per_label_thr = find_best_thresholds_per_label(Global_Model, val_loader, config["device"])
            g_loss, metrics = eval_model(Global_Model, config["device"], val_loader, per_label_thr=per_label_thr)
            history["global_loss"].append(g_loss)
            history["global_metrics"].append(metrics)
            save_json(history, os.path.join(alpha_dir, f"metrics_seed_{config['seed']}.json"))

            torch.save(Global_Model.state_dict(), os.path.join(alpha_dir, f"latest_model_seed_{config['seed']}.pt"))

            if metrics["auc_macro"] > max_auc_macro:
                max_auc_macro = metrics["auc_macro"]
                torch.save(Global_Model.state_dict(), os.path.join(alpha_dir, f"max_auc_seed_{config['seed']}.pt"))
            if metrics["f1_micro"] > max_f1_micro:
                max_f1_micro = metrics["f1_micro"]
                torch.save(Global_Model.state_dict(), os.path.join(alpha_dir, f"max_f1_micro_seed_{config['seed']}.pt"))

    print(f"{algo_name} training complete.")

    # === 7. Final Test Evaluation ===
    _, per_label_thr = find_best_thresholds_per_label(Global_Model, val_loader, config["device"])
    test_loss, test_metrics = eval_model(Global_Model, config["device"], test_loader, per_label_thr=per_label_thr)

    # === 8. Save Results ===
    result_entry = {
        "algorithm": algo_name,
        "alpha": config["alpha"],
        "seed": config["seed"],
        "num_clients": config["num_clients"],
        "clients_per_round": config["clients_per_round"],
        "local_epochs": config["local_epochs"],
        "batch_size": config["batch_size"],
        "lr": config["lr"],
        "F1_micro": round(test_metrics["f1_micro"], 4),
        "F1_macro": round(test_metrics["f1_macro"], 4),
        "AUC_micro": round(test_metrics["auc_micro"], 4),
        "AUC_macro": round(test_metrics["auc_macro"], 4),
        "PR_micro": round(test_metrics.get("prauc_micro", 0.0), 4),
        "PR_macro": round(test_metrics.get("prauc_macro", 0.0), 4),
        "test_loss": round(test_loss, 4)
    }

    summary_path = os.path.join(algo_dir, "summary.csv")
    global_summary_path = os.path.join(config["save_root"], "global_summary.csv")

    df = pd.DataFrame([result_entry])
    if os.path.exists(summary_path):
        df = pd.concat([pd.read_csv(summary_path), df], ignore_index=True)
    df.to_csv(summary_path, index=False)

    if os.path.exists(global_summary_path):
        dfg = pd.read_csv(global_summary_path)
        df = pd.concat([dfg, pd.DataFrame([result_entry])], ignore_index=True)
    df.to_csv(global_summary_path, index=False)

    print(f"Results saved for {algo_name} α={config['alpha']} seed={config['seed']}")


# %%
# === Test Plan Sweep ===

alphas = [0.2, 1.0]
num_clients = [3, 4]
local_epochs = [1, 3]
batch_sizes = [16, 32]
learning_rates = [2e-5, 1e-5]
seeds = [42, 123, 999]
algorithms = ["FedAvg", "FedProx", "SCAFFOLD"]

for algo in algorithms:
    for alpha in alphas:
        for n_clients in num_clients:
            for e in local_epochs:
                for bs in batch_sizes:
                    for lr in learning_rates:
                        for seed in seeds:
                            config = {
                                "algorithm": algo,
                                "alpha": alpha,
                                "num_clients": n_clients,
                                "clients_per_round": max(1, n_clients // 2),
                                "local_epochs": e,
                                "rounds": 100,
                                "batch_size": bs,
                                "lr": lr,
                                "max_seq_len": 256,
                                "n_filters": 21,
                                "window_size": 6,
                                "seed": seed,
                                "device": "cuda" if torch.cuda.is_available() else "cpu",
                                "save_root": "../History/TestPlan"
                            }
                            run_fl_experiment(config)



