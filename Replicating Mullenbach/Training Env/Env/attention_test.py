# %% [markdown]
# # Attention Alignment Notebook
# 
# Goal: Compare **attention behavior** across:
# - Centralized model
# - FedAvg
# - FedProx
# - SCAFFOLD
# 
# We will answer 4 questions:
# 
# 1) Do models focus on the same tokens given the same input **and prediction**?
# 2) Are there cases where models make the same prediction but rely on different tokens?
# 3) How does attention agreement change across label frequency?
# 4) Which federated method is closest to centralized in attention structure?
# 
# Key principle: **Prediction agreement uses each model’s own per-label thresholds** (computed on validation).
# 

# %% [markdown]
# ## Set Up

# %%
# Imports

from torch.utils.data import DataLoader, TensorDataset
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple

import os
import math
from collections import Counter

import matplotlib.pyplot as plt
from IPython.display import HTML, display

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# %%
# Data Loader

def load_data(split: str, batch_size: int = 32, shuffle: bool = False) -> DataLoader:
    X = torch.load(os.path.join("..", "Data", f"X_{split}.pt"))
    Y = torch.load(os.path.join("..", "Data", f"Y_{split}.pt"))
    return DataLoader(
        TensorDataset(X, Y),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True
    )

# %%
# Model Definition

class ConvAttnPool(nn.Module):
    def __init__(
        self,
        table_path: str,
        label_space: int = 50,
        num_of_filters: int = 10,
        kernel_size: int = 3,
        drop_out: float = 0.2
    ):
        super().__init__()
        w2v = Word2Vec.load(table_path)
        vocab_size, embed_d = w2v.wv.vectors.shape

        embed_table = torch.from_numpy(w2v.wv.vectors).float()
        embed_table = torch.cat([embed_table, torch.zeros((1, embed_d))], dim=0)

        self.embed = nn.Embedding.from_pretrained(embeddings=embed_table, padding_idx=vocab_size)
        self.embed_drop = nn.Dropout(p=drop_out)

        self.conv = nn.Conv1d(
            in_channels=embed_d,
            out_channels=num_of_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.U = nn.Linear(num_of_filters, label_space)
        self.final = nn.Linear(num_of_filters, label_space)

        self.embedding_size = embed_d

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(x)                      # (B, L, E)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)                  # (B, E, L)
        x = torch.tanh(self.conv(x).transpose(1, 2))  # (B, L, F)

        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)  # (B, C, L)
        m = alpha.matmul(x)                    # (B, C, F)
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)       # (B, C)
        return y, alpha


def GenerateModel(table_path: str, n_filters: int, window_size: int, label_space: int = 50) -> ConvAttnPool:
    return ConvAttnPool(
        table_path=table_path,
        drop_out=0.2,
        num_of_filters=n_filters,
        kernel_size=window_size,
        label_space=label_space,
    )


# %%
# Config + Token Decoder Setup

config = {
    "batch_size": 32,
    "n_filters": 21,
    "window_size": 6,
}

model_param_path = os.path.join("..", "Model", "processed_full.w2v")

# Recover mapping index -> token, plus PAD index convention
w2v = Word2Vec.load(model_param_path)
idx_to_token = w2v.wv.index_to_key
PAD_INDEX = len(idx_to_token)

def decode_sequence(token_ids_1d: torch.Tensor) -> List[str]:
    toks = []
    for idx in token_ids_1d.detach().cpu().tolist():
        if idx == PAD_INDEX:
            toks.append("<PAD>")
        else:
            toks.append(idx_to_token[idx])
    return toks

print("Vocab size:", len(idx_to_token), "| PAD_INDEX:", PAD_INDEX)


# %%
# Load Models + Datasets

MODEL_PATHS = {
    "Centralized": "../History/models/central_best_attention.pt",
    "FedAvg":      "../History/models/fedavg_c2e3_best_attention.pt",
    "FedProx":     "../History/models/fedprox_c2e3_best_attention.pt",
    "SCAFFOLD":    "../History/models/scaffold_c2e3_best_attention.pt",
}

def load_model(path: str) -> nn.Module:
    model = GenerateModel(
        table_path=model_param_path,
        n_filters=config["n_filters"],
        window_size=config["window_size"],
    ).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

models: Dict[str, nn.Module] = {name: load_model(p) for name, p in MODEL_PATHS.items()}
print("Loaded models:", list(models.keys()))

# Load splits
train_loader = load_data("train", batch_size=config["batch_size"], shuffle=False)
val_loader   = load_data("val",   batch_size=config["batch_size"], shuffle=False)
test_loader  = load_data("test",  batch_size=config["batch_size"], shuffle=False)

# Also keep datasets for random indexing convenience
X_test = torch.load(os.path.join("..", "Data", "X_test.pt"))
Y_test = torch.load(os.path.join("..", "Data", "Y_test.pt"))
test_dataset = TensorDataset(X_test, Y_test)
print("Test samples:", len(test_dataset))


# %% [markdown]
# ## Helper Functions

# %%
# Recomputer Per-Label Thresholds Per Model

@torch.no_grad()
def find_best_thresholds_per_label(model: nn.Module, data_loader: DataLoader, device: torch.device) -> torch.Tensor:
    """
    Returns thresholds tensor of shape (C,) on CPU.
    Threshold grid: 0.05..0.90 step 0.05 (matches your old notebook logic).
    """
    model.eval()
    all_logits, all_y = [], []
    for Xb, yb in data_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        logits, _ = model(Xb)
        all_logits.append(logits)
        all_y.append(yb)

    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)
    probs = torch.sigmoid(logits)

    C = y.shape[1]
    best_thr = torch.zeros(C, device=device)

    for c in range(C):
        best_f1, best_t = -1.0, 0.3
        y_c = y[:, c].long()
        p_c = probs[:, c]
        for t in torch.arange(0.05, 0.95, 0.05, device=device):
            pred = (p_c >= t).long()
            tp = (pred * y_c).sum().item()
            fp = (pred * (1 - y_c)).sum().item()
            fn = ((1 - pred) * y_c).sum().item()
            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            f1   = 2 * prec * rec / (prec + rec + 1e-9)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t.item())
        best_thr[c] = best_t

    return best_thr.detach().cpu()

per_label_thr_by_model: Dict[str, torch.Tensor] = {}
for name, m in models.items():
    per_label_thr_by_model[name] = find_best_thresholds_per_label(m, val_loader, device)

print("Computed per-label thresholds for:", list(per_label_thr_by_model.keys()))
print("Centralized thresholds (first 10):", per_label_thr_by_model["Centralized"][:10])


# %%
# Attention + Logits Extraction

@torch.no_grad()
def get_logits_and_alpha(model: nn.Module, X_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      logits: (B, C) on CPU
      alpha : (B, C, L) on CPU
    """
    X_batch = X_batch.to(device)
    logits, alpha = model(X_batch)
    return logits.detach().cpu(), alpha.detach().cpu()

@torch.no_grad()
def get_label_logits_and_attn(
    model: nn.Module,
    X_batch: torch.Tensor,
    label_idx: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      logits_label: (B,) CPU
      attn_label  : (B, L) CPU
    """
    logits, alpha = get_logits_and_alpha(model, X_batch)
    return logits[:, label_idx], alpha[:, label_idx, :]


# %%
# Padding / Valid Positions (PAD + conv-edge trim)

def get_nonpad_len(x_1d: torch.Tensor, pad_index: int) -> int:
    x = x_1d.detach().cpu()
    pad_pos = (x == pad_index).nonzero(as_tuple=False)
    return int(pad_pos[0].item()) if len(pad_pos) > 0 else int(x.numel())

def get_valid_positions(
    x_1d: torch.Tensor,
    pad_index: int,
    window_size: Optional[int],
) -> np.ndarray:
    """
    Valid positions exclude:
      - PAD region (assumed at end)
      - conv-edge artifacts by trimming half window at each side (optional)
    """
    L = get_nonpad_len(x_1d, pad_index)
    if L <= 0:
        return np.array([], dtype=int)

    left, right = 0, L
    if window_size is not None and window_size > 1:
        half = window_size // 2
        left = min(left + half, L)
        right = max(right - half, left)

    return np.arange(left, right, dtype=int)


# %%
# Similarity Metrics (Cosine + Top-k Jaccard on valid positions)

def cosine_sim_on_valid(att_a_1d: torch.Tensor, att_b_1d: torch.Tensor, valid_pos: np.ndarray) -> float:
    if valid_pos.size == 0:
        return float("nan")
    a = att_a_1d[valid_pos]
    b = att_b_1d[valid_pos]
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

def topk_positions(att_1d: torch.Tensor, valid_pos: np.ndarray, k: int) -> set:
    if valid_pos.size == 0:
        return set()
    att = att_1d.detach().cpu().numpy()
    att_valid = att[valid_pos]
    if att_valid.size == 0:
        return set()
    k_eff = min(k, att_valid.size)
    idx_part = np.argpartition(att_valid, -k_eff)[-k_eff:]
    idx_sorted = idx_part[np.argsort(att_valid[idx_part])[::-1]]
    return set(valid_pos[idx_sorted].tolist())

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# %%
# Prediction Helpers (per-model thresholds + agreement sets)

@torch.no_grad()
def predict_label_for_samples(
    model_name: str,
    X_batch: torch.Tensor,
    label_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      logits: (B,) CPU
      probs : (B,) CPU
      pred  : (B,) bool CPU, using model-specific per-label threshold
    """
    model = models[model_name]
    thr = per_label_thr_by_model[model_name][label_idx].item()

    logits, _ = get_label_logits_and_attn(model, X_batch, label_idx)
    probs = torch.sigmoid(logits)
    pred = (probs >= thr)
    return logits, probs, pred

@torch.no_grad()
def agree_positive_mask(
    model_a: str,
    model_b: str,
    X_batch: torch.Tensor,
    label_idx: int,
) -> torch.Tensor:
    """
    Returns boolean mask (B,) where BOTH models predict label positive using their own thresholds.
    """
    _, _, pred_a = predict_label_for_samples(model_a, X_batch, label_idx)
    _, _, pred_b = predict_label_for_samples(model_b, X_batch, label_idx)
    return (pred_a & pred_b)

@torch.no_grad()
def agree_prediction_mask(
    model_a: str,
    model_b: str,
    X_batch: torch.Tensor,
    label_idx: int,
) -> torch.Tensor:
    """
    Returns boolean mask (B,) where BOTH models have the same binary prediction (pos OR neg).
    """
    _, _, pred_a = predict_label_for_samples(model_a, X_batch, label_idx)
    _, _, pred_b = predict_label_for_samples(model_b, X_batch, label_idx)
    return (pred_a == pred_b)


# %%
# Qualitative Rendering Helpers (with prediction metadata)

def attention_to_html(tokens: List[str], attn_1d: torch.Tensor, max_color: int = 240) -> HTML:
    att = attn_1d.detach().cpu()
    denom = att.max().clamp_min(1e-9)
    att = att / denom

    html = ""
    for tok, w in zip(tokens, att):
        intensity = int(max_color * float(w))
        html += (
            f"<span style='background-color: rgb(255,255,{255-intensity}); "
            f"padding:2px; margin:1px;'>{tok}</span>"
        )
    return HTML(html)

def get_top_tokens(tokens: List[str], attn_1d: torch.Tensor, k: int = 15, skip_pad: bool = True):
    att = attn_1d.detach().cpu().numpy()
    idxs = np.argsort(att)[::-1]
    out = []
    for idx in idxs:
        tok = tokens[idx]
        if skip_pad and tok == "<PAD>":
            continue
        out.append((idx, tok, float(att[idx])))
        if len(out) >= k:
            break
    return out

@torch.no_grad()
def show_attention(
    model_name: str,
    label_idx: int,
    sample_dataset_idx: int,
    top_k: int = 15,
    trim_context: bool = True,
):
    x, y = test_dataset[sample_dataset_idx]
    tokens = decode_sequence(x)

    # logits + attn
    logits_lbl, attn_lbl = get_label_logits_and_attn(models[model_name], x.unsqueeze(0), label_idx)
    logits = logits_lbl[0].item()
    prob = float(torch.sigmoid(logits_lbl[0]).item())
    thr = float(per_label_thr_by_model[model_name][label_idx].item())
    pred = bool(prob >= thr)

    attn_vec = attn_lbl[0]  # (L,)

    # valid positions
    window_size = config["window_size"] if trim_context else None
    valid_pos = get_valid_positions(x, PAD_INDEX, window_size)

    print(f"\nModel={model_name} | label={label_idx} | sample={sample_dataset_idx}")
    print(f"  logit={logits:.4f} | prob={prob:.4f} | thr={thr:.2f} | pred={pred} | GT={int(y[label_idx].item())}")
    print(f"  valid_len={len(valid_pos)} / seq_len={len(tokens)} (trim_context={trim_context})")

    display(attention_to_html(tokens, attn_vec))

    top = get_top_tokens(tokens, attn_vec, k=top_k)
    print(f"Top-{top_k} tokens by attention:")
    for pos, tok, w in top:
        marker = "" if pos in set(valid_pos.tolist()) else "  (trimmed)"
        print(f"  pos={pos:4d} att={w:.4f} tok='{tok}'{marker}")

# %% [markdown]
# ## Questions

# %% [markdown]
# ### 1. Do different models focus on the same tokens given the same input and prediction?
# 
# This helps us assess whether federated models preserve reasoning patterns similar to centralized training.

# %% [markdown]
# ### 2. Are there cases where models make the same prediction but rely on different tokens?
# 
# These cases are particularly important, as they indicate potential differences in learned clinical rationale even when accuracy agrees.

# %% [markdown]
# ### 3. How does attention agreement change across label frequency?
# 
# We are especially interested in whether agreement degrades for rare diagnoses compared to frequent ones.

# %% [markdown]
# ### 4. Which federated methods behave closest to centralized training in terms of attention structure?
# 
# This allows us to compare federated methods beyond predictive performance and look at behavioral alignment.


