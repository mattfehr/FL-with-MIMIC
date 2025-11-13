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

# Custom metrics from existing evaluation.py
from evaluation import all_metrics


# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# %% [markdown]
# ### Client Functions

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



