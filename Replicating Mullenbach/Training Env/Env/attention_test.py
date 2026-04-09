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
from itertools import combinations

import os
import math
from collections import Counter
import csv
import json

import matplotlib.pyplot as plt
from IPython.display import HTML, display

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# %%
# === Result Saving / Loading Utils ===

def save_dict_rows_to_csv(rows: list[dict], filepath: str) -> None:
    """
    Save a list of dictionaries to a CSV file.
    Handles rows with non-identical keys by building the union of keys in order seen.
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


def to_serializable(obj):
    """
    Recursively convert objects into JSON-serializable Python types.
    """
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.detach().cpu().tolist()
    else:
        return obj


def to_serializable_row(row: dict) -> dict:
    """
    Convert one row dictionary into JSON-safe Python scalars/lists.
    """
    return {k: to_serializable(v) for k, v in row.items()}


def save_json(data: dict, filepath: str) -> None:
    """
    Save a Python object/dictionary to JSON after converting unsupported types.
    """
    with open(filepath, mode="w") as f:
        json.dump(to_serializable(data), fp=f, indent=2)

    print(f"Saved JSON: {filepath}")


def load_json(filepath: str) -> dict:
    """
    Load JSON data from a file.
    """
    with open(filepath, mode="r") as f:
        return json.load(f)

# %%
# Load label metadata

DESC_PATH = os.path.join("..", "Data", "ICD9_descriptions")
CODE_50_PATH = os.path.join("..", "Data", "TOP_50_CODES.csv")

def load_icd_metadata(desc_path: str, code_50_path: str):
    """
    Loads:
      - label_idx -> ICD code
      - ICD code -> long description
    Returns two dicts.
    """
    # ICD code -> description
    desc_df = pd.read_csv(desc_path, sep="\t", header=None, names=["CODE", "LONG_TITLE"])
    code_to_desc = dict(zip(desc_df["CODE"].astype(str), desc_df["LONG_TITLE"]))

    # label index -> ICD code (row order matters!)
    top50_df = pd.read_csv(code_50_path, header=None, names=["CODE"])
    idx_to_code = dict(enumerate(top50_df["CODE"].astype(str)))

    return idx_to_code, code_to_desc


IDX_TO_CODE, CODE_TO_DESC = load_icd_metadata(DESC_PATH, CODE_50_PATH)

def label_info(label_idx: int) -> str:
    """
    Returns a readable string for a label index:
      '401.9 – Unspecified essential hypertension'
    """
    code = IDX_TO_CODE.get(label_idx, "UNK")
    desc = CODE_TO_DESC.get(code, "Unknown ICD code")
    return f"{code} – {desc}"


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
    """
    Reverts tokenization (token ids) back into readable strings
    """
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
    "Centralized": "../History/models/central_e300_best_attention_fixed.pt",
    "FedAvg":      "../History/models/fedavg_c2e3_best_attention_fixed.pt",
    "FedProx":     "../History/models/fedprox_c3e3_mu0001_best_attention_fixed.pt",
    "SCAFFOLD":    "../History/models/scaffold_c4e3_lr125_m0_best_attention_fixed.pt",
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


# %%
# === Attention Results Directory Setup ===

ATTN_RESULTS_DIR = os.path.join("..", "History", "attention_results")
os.makedirs(ATTN_RESULTS_DIR, exist_ok=True)

# Main section folders
Q1_DIR = os.path.join(ATTN_RESULTS_DIR, "q1_token_agreement")
Q2_DIR = os.path.join(ATTN_RESULTS_DIR, "q2_divergent_rationales")
Q3_DIR = os.path.join(ATTN_RESULTS_DIR, "q3_label_frequency")
Q4_DIR = os.path.join(ATTN_RESULTS_DIR, "q4_closest_to_centralized")
Q5_DIR = os.path.join(ATTN_RESULTS_DIR, "q5_high_agreement_mislabeling")
PHRASE_DIR = os.path.join(ATTN_RESULTS_DIR, "phrase_level_attention")

for d in [Q1_DIR, Q2_DIR, Q3_DIR, Q4_DIR, Q5_DIR, PHRASE_DIR]:
    os.makedirs(d, exist_ok=True)

# -------------------------
# Q1: token-level agreement
# -------------------------
Q1_RESULTS_CSV = os.path.join(Q1_DIR, "q1_label_pair_results.csv")
Q1_RESULTS_JSON = os.path.join(Q1_DIR, "q1_label_pair_results.json")

Q1_PAIR_SUMMARY_CSV = os.path.join(Q1_DIR, "q1_pair_summary.csv")
Q1_PAIR_SUMMARY_JSON = os.path.join(Q1_DIR, "q1_pair_summary.json")

Q1_CENTRAL_ONLY_CSV = os.path.join(Q1_DIR, "q1_centralized_vs_federated.csv")

Q1_WEIGHTED_COSINE_PLOT = os.path.join(Q1_DIR, "q1_weighted_cosine_centralized_vs_federated.png")
Q1_WEIGHTED_JACCARD_PLOT = os.path.join(Q1_DIR, "q1_weighted_jaccard_centralized_vs_federated.png")

# -------------------------
# Q2: divergent rationales
# -------------------------
Q2_STATS_CSV = os.path.join(Q2_DIR, "q2_label_coverage_stats.csv")
Q2_STATS_JSON = os.path.join(Q2_DIR, "q2_label_coverage_stats.json")

Q2_CASES_CSV = os.path.join(Q2_DIR, "q2_divergent_cases.csv")
Q2_CASES_JSON = os.path.join(Q2_DIR, "q2_divergent_cases.json")

Q2_TOP_CASES_CSV = os.path.join(Q2_DIR, "q2_top_divergent_cases_overall.csv")

# -------------------------
# Q3: label frequency
# -------------------------
Q3_FREQ_CSV = os.path.join(Q3_DIR, "q3_label_frequencies.csv")
Q3_FREQ_JSON = os.path.join(Q3_DIR, "q3_label_frequencies.json")

Q3_MERGED_CSV = os.path.join(Q3_DIR, "q3_frequency_vs_agreement_all_pairs.csv")
Q3_CENTRAL_ONLY_CSV = os.path.join(Q3_DIR, "q3_frequency_vs_agreement_centralized_vs_federated.csv")

Q3_CORR_CSV = os.path.join(Q3_DIR, "q3_spearman_correlations.csv")
Q3_CORR_JSON = os.path.join(Q3_DIR, "q3_spearman_correlations.json")

Q3_BINS_CSV = os.path.join(Q3_DIR, "q3_binned_trends.csv")
Q3_BINS_JSON = os.path.join(Q3_DIR, "q3_binned_trends.json")

Q3_BINNED_COSINE_PLOT = os.path.join(Q3_DIR, "q3_binned_cosine_all_pairs.png")
Q3_BINNED_JACCARD_PLOT = os.path.join(Q3_DIR, "q3_binned_jaccard_all_pairs.png")

# -------------------------
# Q4: closest to centralized
# -------------------------
Q4_RANK_CSV = os.path.join(Q4_DIR, "q4_method_ranking.csv")
Q4_RANK_JSON = os.path.join(Q4_DIR, "q4_method_ranking.json")

Q4_WIN_RATES_CSV = os.path.join(Q4_DIR, "q4_per_label_win_rates.csv")
Q4_WIN_RATES_JSON = os.path.join(Q4_DIR, "q4_per_label_win_rates.json")

Q4_WINNER_DIFF_LABELS_CSV = os.path.join(Q4_DIR, "q4_labels_where_cosine_and_jaccard_winners_differ.csv")

Q4_BOOTSTRAP_CI_CSV = os.path.join(Q4_DIR, "q4_bootstrap_confidence_intervals.csv")
Q4_BOOTSTRAP_CI_JSON = os.path.join(Q4_DIR, "q4_bootstrap_confidence_intervals.json")

Q4_FREQ_BINS_CSV = os.path.join(Q4_DIR, "q4_frequency_bin_breakdown.csv")
Q4_FREQ_BINS_JSON = os.path.join(Q4_DIR, "q4_frequency_bin_breakdown.json")

Q4_COSINE_BAR_PLOT = os.path.join(Q4_DIR, "q4_weighted_cosine_ranked.png")
Q4_JACCARD_BAR_PLOT = os.path.join(Q4_DIR, "q4_weighted_jaccard_ranked.png")
Q4_FREQ_BIN_COSINE_PLOT = os.path.join(Q4_DIR, "q4_frequency_bin_cosine.png")
Q4_FREQ_BIN_JACCARD_PLOT = os.path.join(Q4_DIR, "q4_frequency_bin_jaccard.png")

# -----------------------------------------
# Q5: high agreement + mislabeling analysis
# -----------------------------------------
Q5_ALL_AGREE_POS_CSV = os.path.join(Q5_DIR, "q5_all_pairs_agree_positive_rows.csv")
Q5_ALL_AGREE_POS_JSON = os.path.join(Q5_DIR, "q5_all_pairs_agree_positive_rows.json")

Q5_SUMMARY_CSV = os.path.join(Q5_DIR, "q5_agreement_mislabeling_summary.csv")
Q5_SUMMARY_JSON = os.path.join(Q5_DIR, "q5_agreement_mislabeling_summary.json")

Q5_COS_BINS_CSV = os.path.join(Q5_DIR, "q5_fp_rate_by_cos_quantile.csv")
Q5_JAC_BINS_CSV = os.path.join(Q5_DIR, "q5_fp_rate_by_jaccard_quantile.csv")
Q5_AVG_BINS_CSV = os.path.join(Q5_DIR, "q5_fp_rate_by_avg_agreement_quantile.csv")
Q5_MIN_BINS_CSV = os.path.join(Q5_DIR, "q5_fp_rate_by_min_agreement_quantile.csv")

Q5_COS_FP_PLOT = os.path.join(Q5_DIR, "q5_fp_rate_vs_cos_quantiles.png")
Q5_JAC_FP_PLOT = os.path.join(Q5_DIR, "q5_fp_rate_vs_jaccard_quantiles.png")
Q5_AVG_FP_PLOT = os.path.join(Q5_DIR, "q5_fp_rate_vs_avg_agreement_quantiles.png")
Q5_MIN_FP_PLOT = os.path.join(Q5_DIR, "q5_fp_rate_vs_min_agreement_quantiles.png")

# -------------------------
# Phrase-level attention
# -------------------------
PHRASE_COVERAGE_CSV = os.path.join(PHRASE_DIR, "phrase_similarity_coverage.csv")
PHRASE_COVERAGE_JSON = os.path.join(PHRASE_DIR, "phrase_similarity_coverage.json")

PHRASE_SAMPLE_LEVEL_CSV = os.path.join(PHRASE_DIR, "phrase_similarity_sample_level.csv")
PHRASE_SAMPLE_LEVEL_JSON = os.path.join(PHRASE_DIR, "phrase_similarity_sample_level.json")

PHRASE_SUMMARY_BY_LABEL_CSV = os.path.join(PHRASE_DIR, "phrase_similarity_summary_by_label.csv")
PHRASE_SUMMARY_BY_LABEL_JSON = os.path.join(PHRASE_DIR, "phrase_similarity_summary_by_label.json")

PHRASE_OVERALL_SUMMARY_CSV = os.path.join(PHRASE_DIR, "phrase_similarity_overall_summary.csv")

PHRASE_LOWEST_LABELS_CSV = os.path.join(PHRASE_DIR, "phrase_lowest_similarity_labels.csv")
PHRASE_HIGHEST_LABELS_CSV = os.path.join(PHRASE_DIR, "phrase_highest_similarity_labels.csv")

print("Attention results directories ready:")
for name, path in {
    "ATTN_RESULTS_DIR": ATTN_RESULTS_DIR,
    "Q1_DIR": Q1_DIR,
    "Q2_DIR": Q2_DIR,
    "Q3_DIR": Q3_DIR,
    "Q4_DIR": Q4_DIR,
    "Q5_DIR": Q5_DIR,
    "PHRASE_DIR": PHRASE_DIR,
}.items():
    print(f"  {name}: {path}")

# %% [markdown]
# ## Helper Functions

# %% [markdown]
# ### Function Definitions

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
    Gets all the logits and attention for every label
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
    Focuses to single labels logits and attention
    Returns:
      logits_label: (B,) CPU
      attn_label  : (B, L) CPU
    """
    logits, alpha = get_logits_and_alpha(model, X_batch)
    return logits[:, label_idx], alpha[:, label_idx, :]


# %%
# Padding / Valid Positions (PAD + conv-edge trim)

def get_nonpad_len(x_1d: torch.Tensor, pad_index: int) -> int:
    """
    Finds the length of the nonpadded sequence
    """
    x = x_1d.detach().cpu()
    pad_pos = (x == pad_index).nonzero(as_tuple=False)
    return int(pad_pos[0].item()) if len(pad_pos) > 0 else int(x.numel())

def get_valid_positions(
    x_1d: torch.Tensor,
    pad_index: int,
    window_size: Optional[int],
) -> np.ndarray:
    """
    Finds valid indices of token positions that are safe and meaningful
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

# The masks are needed to filter for certain meaninful samples

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

def get_top_tokens_valid(
    tokens: List[str],
    attn_1d: torch.Tensor,
    valid_pos: np.ndarray,
    k: int = 15,
):
    """
    Return top-k tokens by attention, restricted to valid positions only.
    This matches how cosine/Jaccard are computed.
    """
    if valid_pos.size == 0:
        return []

    att = attn_1d.detach().cpu().numpy()
    att_valid = att[valid_pos]

    k_eff = min(k, att_valid.size)
    idx_part = np.argpartition(att_valid, -k_eff)[-k_eff:]
    idx_sorted = idx_part[np.argsort(att_valid[idx_part])[::-1]]

    out = []
    for i in idx_sorted:
        pos = int(valid_pos[i])
        out.append((pos, tokens[pos], float(att[pos])))

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

    print(f"\nModel={model_name} | label={label_idx} ({label_info(label_idx)}) | sample={sample_dataset_idx}")
    print(f"  logit={logits:.4f} | prob={prob:.4f} | thr={thr:.2f} | pred={pred} | GT={int(y[label_idx].item())}")
    print(f"  valid_len={len(valid_pos)} / seq_len={len(tokens)} (trim_context={trim_context})")

    display(attention_to_html(tokens, attn_vec))

    top = get_top_tokens_valid(tokens, attn_vec, valid_pos, k=top_k)
    print(f"Top-{top_k} tokens by attention (VALID positions only):")
    for pos, tok, w in top:
        print(f"  pos={pos:4d} att={w:.4f} tok='{tok}'")


# %%
def plot_attention_heatmap(
    sample_idx: int,
    label_idx: int,
    model_names: list,
    trim_context: bool = True,
    mode: str = "position",          # "position" (old-style) or "token_zoom"
    max_tokens: int = None,          # for "position": None shows all valid positions; for "token_zoom" acts as width
    start: int = 0,                  # for "token_zoom": start offset into valid positions
    tick_every: int = 10,            # for "token_zoom": label every N tokens
):
    """
    Plot attention heatmaps for multiple models on the same sample+label.

    Modes:
      - mode="position"  : old-style, x-axis is token position (valid positions only). Best for long sequences.
      - mode="token_zoom": zoomed window with token strings on x-axis. Best for local inspection.

    Rows = models
    Cols = token positions (mode="position") or a zoomed token window (mode="token_zoom")
    """
    x, y = test_dataset[sample_idx]
    tokens = decode_sequence(x)

    window_size = config["window_size"] if trim_context else None
    valid_pos = get_valid_positions(x, PAD_INDEX, window_size)

    if len(valid_pos) == 0:
        print("No valid positions to plot (sequence may be empty or fully padded).")
        return

    if mode not in {"position", "token_zoom"}:
        raise ValueError("mode must be 'position' or 'token_zoom'")

    # --- choose which positions to plot ---
    if mode == "position":
        pos = valid_pos
        if max_tokens is not None and len(pos) > max_tokens:
            pos = pos[:max_tokens]
    else:  # token_zoom
        width = 60 if max_tokens is None else int(max_tokens)
        pos = valid_pos[start:start + width]
        if len(pos) == 0:
            print("Token zoom window is empty (check start/max_tokens).")
            return

    # --- build attention matrix (models x positions) ---
    attn_matrix = []
    for mname in model_names:
        _, att = get_label_logits_and_attn(models[mname], x.unsqueeze(0), label_idx)
        attn_matrix.append(att[0][pos].detach().cpu().numpy())
    attn_matrix = np.vstack(attn_matrix)

    # --- plot ---
    if mode == "position":
        # old-style: x is token position
        plt.figure(figsize=(14, 2 + 1.2 * len(model_names)))
        plt.imshow(
            attn_matrix,
            aspect="auto",
            interpolation="nearest",
            extent=[int(pos[0]), int(pos[-1]), len(model_names), 0],  # x-axis in true positions
        )
        plt.colorbar(label="Attention weight")
        plt.yticks(np.arange(len(model_names)) + 0.5, model_names)
        plt.xlabel("Token position (valid positions only)")
        plt.title(f"Attention Heatmap (by position) | sample={sample_idx}, label={label_idx} ({label_info(label_idx)})")
        plt.tight_layout()
        plt.show()

    else:
        # zoomed: x ticks are token strings
        plt.figure(figsize=(min(14, len(pos) * 0.25), 2 + 1.2 * len(model_names)))
        plt.imshow(attn_matrix, aspect="auto", interpolation="nearest")
        plt.colorbar(label="Attention weight")
        plt.yticks(range(len(model_names)), model_names)

        xticks = list(range(0, len(pos), max(1, int(tick_every))))
        xticklabels = [tokens[int(pos[i])] for i in xticks]
        plt.xticks(xticks, xticklabels, rotation=90)

        plt.xlabel("Tokens (zoomed window)")
        plt.title(f"Attention Heatmap (token zoom) | sample={sample_idx}, label={label_idx}")
        plt.tight_layout()
        plt.show()


# %% [markdown]
# ### Sanity tests for helper functions (labels 3 and 7)
# 
# These checks ensure:
# - thresholds exist and look reasonable (0.05–0.90 grid)
# - attention vectors are shaped correctly and sum to ~1 across tokens
# - valid position logic trims PAD and conv edges as intended
# - cosine/Jaccard behave as expected on a few known samples
# - agree-positive masking is actually filtering (not always true/false)
# - qualitative output prints consistent prediction metadata
# 

# %%
# # Test 1: Threshold vectors are valid per model

# TEST_LABELS = [3, 7]

# print("=== Threshold sanity check ===")
# for mname in models.keys():
#     thr = per_label_thr_by_model[mname]
#     print(f"\n{mname}: shape={tuple(thr.shape)}  min={thr.min():.2f}  max={thr.max():.2f}")
#     for lbl in TEST_LABELS:
#         print(f"  label {lbl}: thr={thr[lbl].item():.2f}")

#     # quick assertion-style checks
#     assert thr.ndim == 1, "threshold vector should be 1D"
#     assert (thr >= 0.0).all() and (thr <= 1.0).all(), "thresholds should be in [0,1]"


# %%
# # Test 2: Attention shapes + sum-to-1 checks (softmax)

# print("=== Attention shape + sum-to-1 sanity check ===")

# # pick one arbitrary test sample
# ds_idx = 0
# x, y = test_dataset[ds_idx]
# Xb = x.unsqueeze(0)

# for lbl in TEST_LABELS:
#     print(f"\nSample={ds_idx} | label={lbl}")
#     for mname, model in models.items():
#         logits_lbl, attn_lbl = get_label_logits_and_attn(model, Xb, lbl)
#         att = attn_lbl[0]  # (L,)

#         s = float(att.sum().item())
#         mx = float(att.max().item())
#         print(f"  {mname:11s} attn_shape={tuple(att.shape)} sum={s:.6f} max={mx:.6f}")

#         # should be very close to 1.0 due to softmax
#         assert abs(s - 1.0) < 1e-3, f"{mname} attention does not sum to ~1 (got {s})"


# %%
# # Test 3: valid_positions trimming behaves (PAD + conv edges)

# print("=== valid_positions sanity check ===")

# # pick a sample likely to have PADs (random-ish)
# ds_idx = 123
# x, y = test_dataset[ds_idx]
# nonpad_len = get_nonpad_len(x, PAD_INDEX)

# valid_no_trim = get_valid_positions(x, PAD_INDEX, window_size=None)
# valid_trim = get_valid_positions(x, PAD_INDEX, window_size=config["window_size"])

# print(f"Sample={ds_idx}")
# print("  seq_len:", x.numel())
# print("  nonpad_len:", nonpad_len)
# print("  valid_no_trim:", (valid_no_trim.min() if len(valid_no_trim) else None), "to", (valid_no_trim.max() if len(valid_no_trim) else None), "len=", len(valid_no_trim))
# print("  valid_trim   :", (valid_trim.min() if len(valid_trim) else None), "to", (valid_trim.max() if len(valid_trim) else None), "len=", len(valid_trim))

# # should be subset
# assert len(valid_trim) <= len(valid_no_trim)
# if len(valid_trim) > 0:
#     assert valid_trim[0] >= valid_no_trim[0]
#     assert valid_trim[-1] <= valid_no_trim[-1]


# %%
# # Test 4: cosine/Jaccard self-similarity = 1, and symmetry

# print("=== Similarity metric sanity check ===")

# ds_idx = 0
# x, y = test_dataset[ds_idx]
# Xb = x.unsqueeze(0)
# valid_pos = get_valid_positions(x, PAD_INDEX, window_size=config["window_size"])
# TOP_K_TEST = 15

# for lbl in TEST_LABELS:
#     # pick one model as baseline
#     mname = "Centralized"
#     _, att = get_label_logits_and_attn(models[mname], Xb, lbl)
#     att = att[0]

#     cos_self = cosine_sim_on_valid(att, att, valid_pos)
#     jac_self = jaccard(topk_positions(att, valid_pos, TOP_K_TEST), topk_positions(att, valid_pos, TOP_K_TEST))

#     print(f"\nlabel={lbl} self-sim: cos={cos_self:.6f}  jac={jac_self:.6f}")
#     assert abs(cos_self - 1.0) < 1e-6
#     assert abs(jac_self - 1.0) < 1e-9

#     # symmetry check on a pair
#     a, b = "Centralized", "FedAvg"
#     _, att_a = get_label_logits_and_attn(models[a], Xb, lbl)
#     _, att_b = get_label_logits_and_attn(models[b], Xb, lbl)
#     att_a, att_b = att_a[0], att_b[0]

#     cos_ab = cosine_sim_on_valid(att_a, att_b, valid_pos)
#     cos_ba = cosine_sim_on_valid(att_b, att_a, valid_pos)

#     jac_ab = jaccard(topk_positions(att_a, valid_pos, TOP_K_TEST), topk_positions(att_b, valid_pos, TOP_K_TEST))
#     jac_ba = jaccard(topk_positions(att_b, valid_pos, TOP_K_TEST), topk_positions(att_a, valid_pos, TOP_K_TEST))

#     print(f"  symmetry: cos_ab={cos_ab:.6f} cos_ba={cos_ba:.6f} | jac_ab={jac_ab:.6f} jac_ba={jac_ba:.6f}")
#     assert abs(cos_ab - cos_ba) < 1e-9
#     assert abs(jac_ab - jac_ba) < 1e-12


# %%
# # Test 5: agree-positive mask actually filters (find examples)

# print("=== agree_positive_mask sanity check (find actual kept samples) ===")

# pair = ("Centralized", "FedAvg")
# max_to_find = 5

# for lbl in TEST_LABELS:
#     found = []
#     for ds_idx in range(len(test_dataset)):
#         x, y = test_dataset[ds_idx]
#         Xb = x.unsqueeze(0)
#         if bool(agree_positive_mask(pair[0], pair[1], Xb, lbl).item()):
#             found.append(ds_idx)
#         if len(found) >= max_to_find:
#             break

#     print(f"\nlabel={lbl} | pair={pair[0]} vs {pair[1]} | found agree-positive samples:", found)

#     # It’s possible a label is super rare; don’t hard-assert >0,
#     # but if found, show prediction metadata for the first one.
#     if found:
#         ds0 = found[0]
#         for m in pair:
#             show_attention(m, lbl, ds0, top_k=15, trim_context=True)


# %%
# # Test 6: “old notebook parity” check for labels 3 and 7

# print("=== Parity check: per-sample similarity matrices (labels 3 and 7) ===")

# MODEL_NAMES = list(models.keys())
# TOP_K = 15
# TRIM = True

# # pick one sample index per label where GT=1 (like your old notebook)
# def find_first_gt_positive(label_idx: int, max_scan: int = 5000) -> Optional[int]:
#     for i in range(min(max_scan, len(test_dataset))):
#         _, y = test_dataset[i]
#         if int(y[label_idx].item()) == 1:
#             return i
#     return None

# for lbl in TEST_LABELS:
#     ds_idx = find_first_gt_positive(lbl)
#     print(f"\nLabel={lbl} | first GT-positive sample={ds_idx}")
#     if ds_idx is None:
#         print("  (none found in scan range)")
#         continue

#     x, y = test_dataset[ds_idx]
#     Xb = x.unsqueeze(0)
#     window_size = config["window_size"] if TRIM else None
#     valid_pos = get_valid_positions(x, PAD_INDEX, window_size)

#     # collect attention vectors
#     att_by_model = {}
#     for mname in MODEL_NAMES:
#         _, att = get_label_logits_and_attn(models[mname], Xb, lbl)
#         att_by_model[mname] = att[0]

#     # cosine matrix
#     cosM = np.zeros((len(MODEL_NAMES), len(MODEL_NAMES)))
#     jacM = np.zeros((len(MODEL_NAMES), len(MODEL_NAMES)))

#     for i, a in enumerate(MODEL_NAMES):
#         for j, b in enumerate(MODEL_NAMES):
#             cosM[i, j] = cosine_sim_on_valid(att_by_model[a], att_by_model[b], valid_pos)
#             jacM[i, j] = jaccard(topk_positions(att_by_model[a], valid_pos, TOP_K),
#                                  topk_positions(att_by_model[b], valid_pos, TOP_K))

#     display(pd.DataFrame(cosM, index=MODEL_NAMES, columns=MODEL_NAMES))
#     display(pd.DataFrame(jacM, index=MODEL_NAMES, columns=MODEL_NAMES))

#     # show predictions for context
#     for mname in MODEL_NAMES:
#         logits, probs, pred = predict_label_for_samples(mname, Xb, lbl)
#         thr = per_label_thr_by_model[mname][lbl].item()
#         print(f"  {mname:11s} prob={probs[0].item():.4f} thr={thr:.2f} pred={bool(pred[0].item())} GT={int(y[lbl].item())}")


# %%
# # Test 7: attention heatmap sanity check (agree-positive samples) — shows BOTH modes

# print("=== attention heatmap sanity check (position + token_zoom) ===")

# pair = ("Centralized", "FedAvg")
# max_to_find = 2

# # position-mode controls
# POS_MAX_TOKENS = None     # None shows full valid range; set e.g. 500 if too wide/heavy

# # token-zoom controls
# ZOOM_START = 0
# ZOOM_WIDTH = 60
# ZOOM_TICK_EVERY = 5

# for lbl in TEST_LABELS:
#     found = []

#     # find agree-positive samples
#     for ds_idx in range(len(test_dataset)):
#         x, y = test_dataset[ds_idx]
#         Xb = x.unsqueeze(0)

#         if bool(agree_positive_mask(pair[0], pair[1], Xb, lbl).item()):
#             found.append(ds_idx)

#         if len(found) >= max_to_find:
#             break

#     print(f"\nlabel={lbl} | pair={pair[0]} vs {pair[1]} | found agree-positive samples:", found)

#     if not found:
#         print("  (no agree-positive samples found; skipping heatmaps)")
#         continue

#     for ds_idx in found:
#         print("\n----------------------------------")
#         print(f"Heatmap test | label={lbl} | sample={ds_idx}")

#         # Print prediction metadata first (important context)
#         for m in pair:
#             x0, _ = test_dataset[ds_idx]
#             logits, probs, pred = predict_label_for_samples(m, x0.unsqueeze(0), lbl)
#             thr = per_label_thr_by_model[m][lbl].item()
#             print(f"  {m:11s} | prob={probs[0].item():.4f} | thr={thr:.2f} | pred={bool(pred[0].item())}")

#         # 1) Old-style: by token position (matches original notebook vibe)
#         print("  -> Plot: mode='position'")
#         plot_attention_heatmap(
#             sample_idx=ds_idx,
#             label_idx=lbl,
#             model_names=list(pair),
#             trim_context=True,
#             mode="position",
#             max_tokens=POS_MAX_TOKENS
#         )

#         # 2) Zoomed: token-labeled window (readable)
#         print("  -> Plot: mode='token_zoom'")
#         plot_attention_heatmap(
#             sample_idx=ds_idx,
#             label_idx=lbl,
#             model_names=list(pair),
#             trim_context=True,
#             mode="token_zoom",
#             start=ZOOM_START,
#             max_tokens=ZOOM_WIDTH,      # width of zoom window
#             tick_every=ZOOM_TICK_EVERY
#         )


# %% [markdown]
# ## Precompute Test-Set Outputs for Fast Analysis
#  
# Many of the analysis sections below (especially Q1, Q2, Q5/Q2H, and Phrase-level)
# repeatedly query the same models on the same test samples.
#  
# To avoid redundant forward passes, we precompute each model's outputs on the full
# test set once and cache:
# - logits
# - attention
# - probabilities
# - binary predictions using each model's per-label thresholds
#  
# Downstream sections should use these cached tensors instead of calling the model again
# inside large triple loops.

# %%
# === Precompute model outputs on the full test set ===

@torch.no_grad()
def precompute_test_outputs(
    models: Dict[str, nn.Module],
    data_loader: DataLoader,
    per_label_thr_by_model: Dict[str, torch.Tensor],
    device: torch.device,
):
    """
    Precompute full-test outputs for each model.

    Returns a dict:
        cache[model_name] = {
            "logits": (N, C) CPU tensor,
            "attn":   (N, C, L) CPU tensor,
            "probs":  (N, C) CPU tensor,
            "preds":  (N, C) CPU bool tensor,
        }
    """
    cache = {}

    for model_name, model in models.items():
        model.eval()
        all_logits = []
        all_attn = []

        for Xb, _ in tqdm(data_loader, desc=f"Precompute {model_name}"):
            Xb = Xb.to(device)
            logits, attn = model(Xb)

            all_logits.append(logits.detach().cpu())
            all_attn.append(attn.detach().cpu())

        logits_all = torch.cat(all_logits, dim=0)   # (N, C)
        attn_all = torch.cat(all_attn, dim=0)       # (N, C, L)
        probs_all = torch.sigmoid(logits_all)       # (N, C)

        thr = per_label_thr_by_model[model_name].cpu().unsqueeze(0)   # (1, C)
        preds_all = probs_all >= thr                                   # (N, C), bool

        cache[model_name] = {
            "logits": logits_all,
            "attn": attn_all,
            "probs": probs_all,
            "preds": preds_all,
        }

        print(
            f"{model_name:11s} | "
            f"logits={tuple(logits_all.shape)} | "
            f"attn={tuple(attn_all.shape)} | "
            f"preds={tuple(preds_all.shape)}"
        )

    return cache


PRECOMP = precompute_test_outputs(
    models=models,
    data_loader=test_loader,
    per_label_thr_by_model=per_label_thr_by_model,
    device=device,
)

print("\nPrecompute complete.")
print("Cached models:", list(PRECOMP.keys()))

# %%
# === Convenience aliases ===

LOGITS_ALL = {m: PRECOMP[m]["logits"] for m in PRECOMP}
ATTN_ALL   = {m: PRECOMP[m]["attn"]   for m in PRECOMP}
PROBS_ALL  = {m: PRECOMP[m]["probs"]  for m in PRECOMP}
PREDS_ALL  = {m: PRECOMP[m]["preds"]  for m in PRECOMP}

N_TEST = next(iter(LOGITS_ALL.values())).shape[0]
N_LABELS = next(iter(LOGITS_ALL.values())).shape[1]

print(f"N_TEST={N_TEST}, N_LABELS={N_LABELS}")

# %%
# === Cached helper functions ===

def get_cached_logits_and_attn(
    model_name: str,
    sample_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      logits: (C,) CPU
      attn  : (C, L) CPU
    """
    return LOGITS_ALL[model_name][sample_idx], ATTN_ALL[model_name][sample_idx]


def get_cached_label_logits_and_attn(
    model_name: str,
    sample_idx: int,
    label_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      logits_label: scalar CPU tensor
      attn_label  : (L,) CPU tensor
    """
    logits = LOGITS_ALL[model_name][sample_idx, label_idx]
    attn = ATTN_ALL[model_name][sample_idx, label_idx]
    return logits, attn


def predict_label_for_sample_cached(
    model_name: str,
    sample_idx: int,
    label_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Cached replacement for per-sample prediction lookup.

    Returns:
      logits: scalar CPU tensor
      probs : scalar CPU tensor
      pred  : scalar CPU bool tensor
    """
    logits = LOGITS_ALL[model_name][sample_idx, label_idx]
    probs = PROBS_ALL[model_name][sample_idx, label_idx]
    pred = PREDS_ALL[model_name][sample_idx, label_idx]
    return logits, probs, pred


def agree_positive_mask_cached(
    model_a: str,
    model_b: str,
    label_idx: int,
) -> torch.Tensor:
    """
    Returns boolean mask (N_TEST,) where BOTH models predict label positive.
    """
    return PREDS_ALL[model_a][:, label_idx] & PREDS_ALL[model_b][:, label_idx]


def agree_prediction_mask_cached(
    model_a: str,
    model_b: str,
    label_idx: int,
) -> torch.Tensor:
    """
    Returns boolean mask (N_TEST,) where BOTH models have the same binary prediction.
    """
    return PREDS_ALL[model_a][:, label_idx] == PREDS_ALL[model_b][:, label_idx]


def get_attn_vector_cached(
    model_name: str,
    sample_idx: int,
    label_idx: int,
) -> torch.Tensor:
    """
    Returns attention vector (L,) on CPU.
    """
    return ATTN_ALL[model_name][sample_idx, label_idx]

# %%
# === Optional: precompute pairwise agreement masks once ===

MODEL_NAMES = list(models.keys())
MODEL_PAIRS = [(a, b) for i, a in enumerate(MODEL_NAMES) for b in MODEL_NAMES[i+1:]]

AGREE_POS_MASKS = {}
AGREE_PRED_MASKS = {}

for a, b in MODEL_PAIRS:
    AGREE_POS_MASKS[(a, b)] = {}
    AGREE_PRED_MASKS[(a, b)] = {}

    for lbl in range(N_LABELS):
        AGREE_POS_MASKS[(a, b)][lbl] = agree_positive_mask_cached(a, b, lbl)
        AGREE_PRED_MASKS[(a, b)][lbl] = agree_prediction_mask_cached(a, b, lbl)

print("Precomputed pairwise agreement masks.")
print("Pairs:", MODEL_PAIRS)

# %%
# === Helper to get candidate indices quickly from cached masks ===

def get_agree_indices_cached(
    model_a: str,
    model_b: str,
    label_idx: int,
    mode: str = "agree_positive",
) -> List[int]:
    """
    Returns dataset indices satisfying a cached pairwise condition.
    """
    if mode == "agree_positive":
        mask = AGREE_POS_MASKS[(model_a, model_b)][label_idx]
    elif mode == "agree_prediction":
        mask = AGREE_PRED_MASKS[(model_a, model_b)][label_idx]
    else:
        raise ValueError("mode must be 'agree_positive' or 'agree_prediction'")

    return torch.nonzero(mask, as_tuple=False).squeeze(1).tolist()

# %% [markdown]
# ### Note on cached analysis
#  
# From this point onward, large-scale analysis sections should use the cached tensors:
# - `LOGITS_ALL`
# - `ATTN_ALL`
# - `PROBS_ALL`
# - `PREDS_ALL`
# - `AGREE_POS_MASKS`
# - `AGREE_PRED_MASKS`
#  
# This avoids repeated model inference inside label × sample × pair loops.

# %%
# === Sanity check for cache correctness ===

print("=== Precompute cache sanity check ===")

test_model = "Centralized"
test_sample = 0
test_label = 3

# cached
logit_c, attn_c = get_cached_label_logits_and_attn(test_model, test_sample, test_label)
prob_c = PROBS_ALL[test_model][test_sample, test_label]
pred_c = PREDS_ALL[test_model][test_sample, test_label]

print(f"Model={test_model} | sample={test_sample} | label={test_label}")
print(f"  cached logit={float(logit_c):.6f}")
print(f"  cached prob ={float(prob_c):.6f}")
print(f"  cached pred ={bool(pred_c)}")
print(f"  attn shape  ={tuple(attn_c.shape)} | sum={float(attn_c.sum()):.6f}")

assert abs(float(attn_c.sum()) - 1.0) < 1e-3, "Cached attention should sum to ~1."
assert LOGITS_ALL[test_model].shape[0] == len(test_dataset), "Cached rows should match test dataset size."

print("Cache sanity check passed.")

# %% [markdown]
# ## Questions

# %% [markdown]
# ### 1. Do different models focus on the same tokens given the same input and prediction?
# 
# This helps us assess whether federated models preserve reasoning patterns similar to centralized training.
# 
# We measure attention agreement **only on samples where two models both predict the label as positive**
# (using each model's own per-label threshold computed on validation). We do specifically positive predictions because negative predictions 
# for labels have attention that is diffuse and low magnitude. So there is no semantic rationale to compare compared to positive predictions
# where attention is actually being used as rationale and justification.
# 
# Metrics:
# - Cosine similarity on attention vectors (valid positions only: no PAD + conv-edge trimmed)
# - Top-k Jaccard overlap on most-attended token positions (valid positions only)
# 
# We also report **coverage**: how many samples survive the agree-positive filter.
# 
# Interpretation
# - High cosine + high Jaccard@K → models concentrate attention on similar token positions.
# - High cosine but low Jaccard → attention mass is spread similarly but top tokens differ.
# - Low cosine + low Jaccard → models rely on different parts of the sequence (different rationale patterns).
# 
# Always check `n_kept`:
# - If `n_kept` is small for a label/pair, treat that label result as noisy.

# %%
# Q1 CONFIG

Q1_LABELS = list(range(50))   # or [3, 7] to start small
Q1_PAIR_MODE = "agree_positive"   # "agree_positive" (recommended) or "agree_prediction"
Q1_TOP_K = 15
Q1_TRIM_CONTEXT = True

# candidate pool options:
Q1_CANDIDATE_POOL = "all_test"     # "all_test" or "gt_positive"

# speed/coverage knobs
Q1_MAX_SAMPLES_PER_LABEL = None    # None = no cap, or set e.g. 300 to speed up
Q1_PROGRESS = True

MODEL_NAMES = list(models.keys())
Q1_PAIRS = [(a, b) for i, a in enumerate(MODEL_NAMES) for b in MODEL_NAMES[i+1:]]

print("Q1_LABELS:", (Q1_LABELS[:10], "...") if len(Q1_LABELS) > 10 else Q1_LABELS)
print("Q1_PAIRS:", Q1_PAIRS)

# %%
def get_candidate_indices_for_label(label_idx: int) -> List[int]:
    """
    Choose candidate dataset indices to test for a label.
    """
    if Q1_CANDIDATE_POOL == "all_test":
        idxs = list(range(len(test_dataset)))
        return idxs[:Q1_MAX_SAMPLES_PER_LABEL] if Q1_MAX_SAMPLES_PER_LABEL else idxs

    if Q1_CANDIDATE_POOL == "gt_positive":
        idxs = torch.nonzero(Y_test[:, label_idx] == 1, as_tuple=False).squeeze(1).tolist()
        return idxs[:Q1_MAX_SAMPLES_PER_LABEL] if Q1_MAX_SAMPLES_PER_LABEL else idxs

    raise ValueError(f"Unknown Q1_CANDIDATE_POOL={Q1_CANDIDATE_POOL}")


def q1_pair_mask_cached_for_label(model_a: str, model_b: str, label_idx: int) -> torch.Tensor:
    """
    Returns cached pairwise mask over the full test set for this label.
    """
    if Q1_PAIR_MODE == "agree_positive":
        return AGREE_POS_MASKS[(model_a, model_b)][label_idx]
    elif Q1_PAIR_MODE == "agree_prediction":
        return AGREE_PRED_MASKS[(model_a, model_b)][label_idx]
    else:
        raise ValueError(f"Unknown Q1_PAIR_MODE={Q1_PAIR_MODE}")

# %%
# Q1 RUN (cached)

q1_rows = []

label_iter = tqdm(Q1_LABELS, desc="Q1 labels") if Q1_PROGRESS else Q1_LABELS
for label_idx in label_iter:
    cand_idxs = get_candidate_indices_for_label(label_idx)
    cand_idx_set = set(cand_idxs)

    for a, b in Q1_PAIRS:
        cos_vals: List[float] = []
        jac_vals: List[float] = []
        kept = 0
        kept_gt_pos = 0
        kept_gt_neg = 0

        # cached pairwise mask over all test samples
        pair_mask_full = q1_pair_mask_cached_for_label(a, b, label_idx)

        # restrict to candidate pool
        kept_idxs = torch.nonzero(pair_mask_full, as_tuple=False).squeeze(1).tolist()
        if Q1_CANDIDATE_POOL != "all_test" or Q1_MAX_SAMPLES_PER_LABEL is not None:
            kept_idxs = [i for i in kept_idxs if i in cand_idx_set]

        for ds_idx in kept_idxs:
            x, y = test_dataset[ds_idx]

            kept += 1
            if int(y[label_idx].item()) == 1:
                kept_gt_pos += 1
            else:
                kept_gt_neg += 1

            window_size = config["window_size"] if Q1_TRIM_CONTEXT else None
            valid_pos = get_valid_positions(x, PAD_INDEX, window_size)

            # cached attention vectors
            att_a_1d = ATTN_ALL[a][ds_idx, label_idx]
            att_b_1d = ATTN_ALL[b][ds_idx, label_idx]

            cos_vals.append(cosine_sim_on_valid(att_a_1d, att_b_1d, valid_pos))

            top_a = topk_positions(att_a_1d, valid_pos, Q1_TOP_K)
            top_b = topk_positions(att_b_1d, valid_pos, Q1_TOP_K)
            jac_vals.append(jaccard(top_a, top_b))

        q1_rows.append({
            "label": label_idx,
            "pair": f"{a} vs {b}",
            "candidate_pool": Q1_CANDIDATE_POOL,
            "pair_mode": Q1_PAIR_MODE,
            "top_k": Q1_TOP_K,
            "trim_context": Q1_TRIM_CONTEXT,
            "n_candidates": len(cand_idxs),
            "n_kept": kept,
            "kept_gt_pos": kept_gt_pos,
            "kept_gt_neg": kept_gt_neg,
            "cos_mean": float(np.nanmean(cos_vals)) if cos_vals else np.nan,
            "cos_median": float(np.nanmedian(cos_vals)) if cos_vals else np.nan,
            "cos_std": float(np.nanstd(cos_vals)) if cos_vals else np.nan,
            "jac_mean": float(np.nanmean(jac_vals)) if jac_vals else np.nan,
            "jac_median": float(np.nanmedian(jac_vals)) if jac_vals else np.nan,
            "jac_std": float(np.nanstd(jac_vals)) if jac_vals else np.nan,
        })

q1_df = pd.DataFrame(q1_rows)
print("Done. Rows:", len(q1_df))
display(q1_df.head())

# %%
# Q1: View coverage + agreement per label (sorted)

q1_sorted = q1_df.sort_values(["n_kept", "jac_mean"], ascending=[False, False])
display(q1_sorted.head(30))

# If you want to only see central-vs-fed comparisons:
central_vs = q1_df[q1_df["pair"].str.contains("Centralized")].copy()
display(central_vs.sort_values(["n_kept", "jac_mean"], ascending=[False, False]).head(30))

# %%
# Q1: Aggregate summary per pair across labels (weighted by coverage)

def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = ~np.isnan(values)
    values = values[mask]
    weights = weights[mask]
    if values.size == 0:
        return float("nan")
    if weights.sum() <= 0:
        return float("nan")
    return float((values * weights).sum() / weights.sum())

pair_summaries = []
for pair, sub in q1_df.groupby("pair"):
    w = sub["n_kept"].to_numpy(dtype=float)

    pair_summaries.append({
        "pair": pair,
        "labels_covered": int(sub["n_kept"].gt(0).sum()),
        "total_kept": int(sub["n_kept"].sum()),
        "cos_mean_weighted": weighted_mean(sub["cos_mean"].to_numpy(dtype=float), w),
        "jac_mean_weighted": weighted_mean(sub["jac_mean"].to_numpy(dtype=float), w),
        "cos_median_unweighted": float(np.nanmedian(sub["cos_median"].to_numpy(dtype=float))),
        "jac_median_unweighted": float(np.nanmedian(sub["jac_median"].to_numpy(dtype=float))),
    })

q1_pair_summary = pd.DataFrame(pair_summaries).sort_values(
    ["jac_mean_weighted", "cos_mean_weighted"], ascending=False
)
display(q1_pair_summary)

# %%
# Q1: Save tabular results

q1_df.to_csv(Q1_RESULTS_CSV, index=False)
save_json(
    {"rows": [to_serializable_row(r) for r in q1_df.to_dict(orient="records")]},
    Q1_RESULTS_JSON
)

q1_pair_summary.to_csv(Q1_PAIR_SUMMARY_CSV, index=False)
save_json(
    {"rows": [to_serializable_row(r) for r in q1_pair_summary.to_dict(orient="records")]},
    Q1_PAIR_SUMMARY_JSON
)

central_vs.to_csv(Q1_CENTRAL_ONLY_CSV, index=False)

print("Saved Q1 results:")
print(" ", Q1_RESULTS_CSV)
print(" ", Q1_RESULTS_JSON)
print(" ", Q1_PAIR_SUMMARY_CSV)
print(" ", Q1_PAIR_SUMMARY_JSON)
print(" ", Q1_CENTRAL_ONLY_CSV)

# %%
# Q1: Bar plots for Centralized vs Federated models

central_pairs = q1_pair_summary[
    q1_pair_summary["pair"].str.contains("Centralized")
]

if len(central_pairs) == 0:
    print("No Centralized pairs found in summary (unexpected if Centralized model is loaded).")
else:
    # --- Weighted Cosine ---
    plt.figure()
    plt.bar(
        central_pairs["pair"],
        central_pairs["cos_mean_weighted"]
    )
    plt.title("Q1: Weighted Cosine Similarity (Centralized vs Federated)")
    plt.ylabel("Weighted cosine similarity")
    plt.xticks(rotation=30, ha="right")

    for i, v in enumerate(central_pairs["cos_mean_weighted"]):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(Q1_WEIGHTED_COSINE_PLOT, dpi=300, bbox_inches="tight")
    plt.show()

    # --- Weighted Jaccard ---
    plt.figure()
    plt.bar(
        central_pairs["pair"],
        central_pairs["jac_mean_weighted"]
    )
    plt.title("Q1: Weighted Jaccard@15 (Centralized vs Federated)")
    plt.ylabel("Weighted Jaccard")
    plt.xticks(rotation=30, ha="right")

    for i, v in enumerate(central_pairs["jac_mean_weighted"]):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(Q1_WEIGHTED_JACCARD_PLOT, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved Q1 plots:")
    print(" ", Q1_WEIGHTED_COSINE_PLOT)
    print(" ", Q1_WEIGHTED_JACCARD_PLOT)

# %% [markdown]
# ### 2. Are there cases where models make the same prediction but rely on different tokens?
# 
# These cases are particularly important, as they indicate potential differences in learned clinical rationale even when accuracy agrees.
# 
# We search for **disagreement-in-rationale** cases:
# - Restrict to samples where both models predict the label positive (**agree-positive**)
# - Score attention similarity on valid positions using:
#   - cosine similarity (lower = more different)
#   - Jaccard@K overlap (lower = different top tokens)
# - Return the most divergent samples and inspect with `show_attention()`

# %%
# Q2 CONFIG

Q2_LABELS = list(range(50))         # or start with [3, 7, 20]
Q2_PAIR = ("Centralized", "FedAvg") # change to ("Centralized","FedProx"), ("Centralized","SCAFFOLD"), etc.

Q2_TOP_K = 15
Q2_TRIM_CONTEXT = True
Q2_CANDIDATE_POOL = "all_test"      # "all_test" or "gt_positive"
Q2_MAX_SCAN_PER_LABEL = None        # None or e.g. 500 for speed
Q2_KEEP_TOP_N_PER_LABEL = 5         # keep top divergent samples per label
Q2_MIN_KEPT_REQUIRED = 30           # skip labels with low coverage (avoid noisy results)

print("Q2_LABELS:", (Q2_LABELS[:10], "...") if len(Q2_LABELS) > 10 else Q2_LABELS)
print("Q2_PAIR:", Q2_PAIR)

# %%
# Helper: candidates for Q2 (cached-friendly)

def q2_candidate_indices_for_label(label_idx: int) -> List[int]:
    if Q2_CANDIDATE_POOL == "all_test":
        idxs = list(range(len(test_dataset)))
        return idxs[:Q2_MAX_SCAN_PER_LABEL] if Q2_MAX_SCAN_PER_LABEL else idxs

    if Q2_CANDIDATE_POOL == "gt_positive":
        idxs = torch.nonzero(Y_test[:, label_idx] == 1, as_tuple=False).squeeze(1).tolist()
        return idxs[:Q2_MAX_SCAN_PER_LABEL] if Q2_MAX_SCAN_PER_LABEL else idxs

    raise ValueError(f"Unknown Q2_CANDIDATE_POOL={Q2_CANDIDATE_POOL}")

# %%
# Q2: Compute per-sample divergence for a given label + pair using cached outputs

def q2_find_divergent_samples_for_label(
    label_idx: int,
    model_a: str,
    model_b: str,
    top_k: int,
    trim_context: bool,
    candidate_indices: List[int],
) -> pd.DataFrame:
    rows = []

    candidate_set = set(candidate_indices)
    kept_idxs = torch.nonzero(
        AGREE_POS_MASKS[(model_a, model_b)][label_idx],
        as_tuple=False
    ).squeeze(1).tolist()

    if Q2_CANDIDATE_POOL != "all_test" or Q2_MAX_SCAN_PER_LABEL is not None:
        kept_idxs = [i for i in kept_idxs if i in candidate_set]

    for ds_idx in kept_idxs:
        x, y = test_dataset[ds_idx]

        window_size = config["window_size"] if trim_context else None
        valid_pos = get_valid_positions(x, PAD_INDEX, window_size)

        att_a_1d = ATTN_ALL[model_a][ds_idx, label_idx]
        att_b_1d = ATTN_ALL[model_b][ds_idx, label_idx]

        cos = cosine_sim_on_valid(att_a_1d, att_b_1d, valid_pos)
        jac = jaccard(
            topk_positions(att_a_1d, valid_pos, top_k),
            topk_positions(att_b_1d, valid_pos, top_k),
        )

        prob_a = float(PROBS_ALL[model_a][ds_idx, label_idx].item())
        prob_b = float(PROBS_ALL[model_b][ds_idx, label_idx].item())
        thr_a = float(per_label_thr_by_model[model_a][label_idx].item())
        thr_b = float(per_label_thr_by_model[model_b][label_idx].item())

        rows.append({
            "label": label_idx,
            "sample": ds_idx,
            "pair": f"{model_a} vs {model_b}",
            "gt": int(y[label_idx].item()),
            "cos": float(cos),
            "jac": float(jac),
            "prob_a": prob_a,
            "thr_a": thr_a,
            "prob_b": prob_b,
            "thr_b": thr_b,
            "valid_len": int(len(valid_pos)),
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(["cos", "jac"], ascending=[True, True]).reset_index(drop=True)

    df.attrs["n_kept"] = len(kept_idxs)
    df.attrs["n_scanned"] = len(candidate_indices)
    return df

# %%
# Q2 RUN: collect top divergent examples per label (single chosen pair)

model_a, model_b = Q2_PAIR

q2_all = []
q2_label_stats = []

for label_idx in tqdm(Q2_LABELS, desc="Q2 labels"):
    cand_idxs = q2_candidate_indices_for_label(label_idx)

    df = q2_find_divergent_samples_for_label(
        label_idx=label_idx,
        model_a=model_a,
        model_b=model_b,
        top_k=Q2_TOP_K,
        trim_context=Q2_TRIM_CONTEXT,
        candidate_indices=cand_idxs,
    )

    n_kept = df.attrs.get("n_kept", 0)
    q2_label_stats.append({
        "label": label_idx,
        "pair": f"{model_a} vs {model_b}",
        "n_scanned": df.attrs.get("n_scanned", len(cand_idxs)),
        "n_agree_pos": n_kept,
        "n_returned": int(min(Q2_KEEP_TOP_N_PER_LABEL, len(df))),
        "cos_min": float(df["cos"].min()) if len(df) else np.nan,
        "jac_min": float(df["jac"].min()) if len(df) else np.nan,
    })

    if n_kept < Q2_MIN_KEPT_REQUIRED:
        continue

    q2_all.append(df.head(Q2_KEEP_TOP_N_PER_LABEL))

q2_label_stats_df = pd.DataFrame(q2_label_stats).sort_values("n_agree_pos", ascending=False)
display(q2_label_stats_df.head(20))

q2_cases_df = pd.concat(q2_all, ignore_index=True) if len(q2_all) else pd.DataFrame()
print("Total Q2 cases collected:", len(q2_cases_df))
display(q2_cases_df.head(20))

# %%
# Q2: Find the most divergent cases overall (across labels)

if len(q2_cases_df) == 0:
    print("No Q2 cases found (check Q2_MIN_KEPT_REQUIRED / pair / candidate pool).")
    q2_most_divergent = pd.DataFrame()
else:
    q2_most_divergent = q2_cases_df.sort_values(["cos", "jac"], ascending=[True, True]).head(30).reset_index(drop=True)
    display(q2_most_divergent)

# %%
# Q2: Save single-pair results

q2_label_stats_df.to_csv(Q2_STATS_CSV, index=False)
save_json(
    {"rows": [to_serializable_row(r) for r in q2_label_stats_df.to_dict(orient="records")]},
    Q2_STATS_JSON
)

q2_cases_df.to_csv(Q2_CASES_CSV, index=False)
save_json(
    {"rows": [to_serializable_row(r) for r in q2_cases_df.to_dict(orient="records")]},
    Q2_CASES_JSON
)

if len(q2_most_divergent) > 0:
    q2_most_divergent.to_csv(Q2_TOP_CASES_CSV, index=False)

print("Saved Q2 single-pair results:")
print(" ", Q2_STATS_CSV)
print(" ", Q2_STATS_JSON)
print(" ", Q2_CASES_CSV)
print(" ", Q2_CASES_JSON)
if len(q2_most_divergent) > 0:
    print(" ", Q2_TOP_CASES_CSV)

# %%
# Q2: Qualitative inspection helper (side-by-side attention + metadata)

def q2_inspect_case(
    row: pd.Series,
    top_k: int = 15,
    trim_context: bool = True,
    do_position_heatmap: bool = True,
):
    """
    For one Q2 case row, show:
      - metadata
      - show_attention for each model (includes VALID-only top-k tokens)
      - heatmap by position (global view)
    """
    label_idx = int(row["label"])
    ds_idx = int(row["sample"])
    a, b = row["pair"].split(" vs ")

    print("\n==============================")
    print(f"Q2 case | label={label_idx} ({label_info(label_idx)}) | sample={ds_idx} | pair={a} vs {b}")
    print(f"  cos={float(row['cos']):.4f} | jac={float(row['jac']):.4f} | GT={int(row['gt'])}")
    print(f"  {a}: prob={float(row['prob_a']):.4f} thr={float(row['thr_a']):.2f}")
    print(f"  {b}: prob={float(row['prob_b']):.4f} thr={float(row['thr_b']):.2f}")
    print(f"  valid_len={int(row.get('valid_len', -1))}")
    print("==============================")

    # qualitative helpers can stay as-is
    show_attention(a, label_idx, ds_idx, top_k=top_k, trim_context=trim_context)
    show_attention(b, label_idx, ds_idx, top_k=top_k, trim_context=trim_context)

    if do_position_heatmap:
        plot_attention_heatmap(
            sample_idx=ds_idx,
            label_idx=label_idx,
            model_names=[a, b],
            trim_context=trim_context,
            mode="position",
            max_tokens=None,
        )

# %%
# Q2: Run for all pairs and produce top-divergent tables per pair (cached)

MODEL_NAMES = list(models.keys())

Q2_PAIRS = [
    (a, b)
    for i, a in enumerate(MODEL_NAMES)
    for b in MODEL_NAMES[i+1:]
]

print("Q2_PAIRS:", Q2_PAIRS)

Q2_LABELS = list(range(50))
Q2_TOP_K = 15
Q2_TRIM_CONTEXT = True
Q2_CANDIDATE_POOL = "all_test"       # "all_test" or "gt_positive"
Q2_MAX_SCAN_PER_LABEL = None
Q2_KEEP_TOP_N_PER_LABEL = 5
Q2_MIN_KEPT_REQUIRED = 30

# qualitative selection
Q2_SHOW_QUAL = True
Q2_EXAMPLE_IDXS = [0, 10]            # extreme + representative rank
Q2_TOP_OVERALL_N = 30                # pool to pick from

def q2_run_for_pair(model_a: str, model_b: str):
    q2_all = []
    q2_label_stats = []

    for label_idx in tqdm(Q2_LABELS, desc=f"Q2 labels ({model_a} vs {model_b})"):
        cand_idxs = q2_candidate_indices_for_label(label_idx)

        df = q2_find_divergent_samples_for_label(
            label_idx=label_idx,
            model_a=model_a,
            model_b=model_b,
            top_k=Q2_TOP_K,
            trim_context=Q2_TRIM_CONTEXT,
            candidate_indices=cand_idxs,
        )

        n_kept = df.attrs.get("n_kept", 0)
        q2_label_stats.append({
            "pair": f"{model_a} vs {model_b}",
            "label": label_idx,
            "n_scanned": df.attrs.get("n_scanned", len(cand_idxs)),
            "n_agree_pos": n_kept,
            "n_returned": int(min(Q2_KEEP_TOP_N_PER_LABEL, len(df))),
            "cos_min": float(df["cos"].min()) if len(df) else np.nan,
            "jac_min": float(df["jac"].min()) if len(df) else np.nan,
        })

        if n_kept < Q2_MIN_KEPT_REQUIRED:
            continue

        q2_all.append(df.head(Q2_KEEP_TOP_N_PER_LABEL))

    stats_df = (
        pd.DataFrame(q2_label_stats)
          .sort_values(["n_agree_pos", "cos_min"], ascending=[False, True])
          .reset_index(drop=True)
    )
    cases_df = pd.concat(q2_all, ignore_index=True) if len(q2_all) else pd.DataFrame()
    return stats_df, cases_df


def pick_3_examples(top_overall: pd.DataFrame) -> list[pd.Series]:
    """
    Pick 3 examples:
      1) extreme: rank 0
      2) representative: rank ~10 (or last)
      3) different-label: first row in top_overall with a new label
    """
    if top_overall is None or len(top_overall) == 0:
        return []

    picks = []

    r0 = top_overall.iloc[0]
    picks.append(r0)

    rep_idx = min(Q2_EXAMPLE_IDXS[1], len(top_overall) - 1)
    r1 = top_overall.iloc[rep_idx]
    picks.append(r1)

    used = {int(r0["label"]), int(r1["label"])}
    r2 = None
    for _, row in top_overall.iterrows():
        if int(row["label"]) not in used:
            r2 = row
            break
    if r2 is not None:
        picks.append(r2)

    return picks[:3]


q2_pair_results = {}  # pair -> {"stats": df, "cases": df, "top_overall": df}

for a, b in Q2_PAIRS:
    stats_df, cases_df = q2_run_for_pair(a, b)

    if len(cases_df) > 0:
        top_overall = (
            cases_df.sort_values(["cos", "jac"], ascending=[True, True])
                    .head(Q2_TOP_OVERALL_N)
                    .reset_index(drop=True)
        )
    else:
        top_overall = pd.DataFrame()

    q2_pair_results[(a, b)] = {
        "stats": stats_df,
        "cases": cases_df,
        "top_overall": top_overall,
    }

    print("\n==============================")
    print(f"PAIR: {a} vs {b}")
    print("Label coverage (top 10 by agree-positive):")
    display(stats_df.head(10))
    print(f"Most divergent cases overall (top {Q2_TOP_OVERALL_N}):")
    display(top_overall)

    if Q2_SHOW_QUAL and len(top_overall) > 0:
        examples = pick_3_examples(top_overall)
        for ex_row in examples:
            q2_inspect_case(
                ex_row,
                top_k=Q2_TOP_K,
                trim_context=Q2_TRIM_CONTEXT,
                do_position_heatmap=True
            )

# %%
# Q2: Save all-pairs results into the same folder (combined files)

q2_all_stats_df = pd.concat(
    [v["stats"] for v in q2_pair_results.values() if v["stats"] is not None and len(v["stats"]) > 0],
    ignore_index=True
) if len(q2_pair_results) > 0 else pd.DataFrame()

q2_all_cases_df = pd.concat(
    [v["cases"] for v in q2_pair_results.values() if v["cases"] is not None and len(v["cases"]) > 0],
    ignore_index=True
) if len(q2_pair_results) > 0 else pd.DataFrame()

q2_all_top_df = pd.concat(
    [v["top_overall"] for v in q2_pair_results.values() if v["top_overall"] is not None and len(v["top_overall"]) > 0],
    ignore_index=True
) if len(q2_pair_results) > 0 else pd.DataFrame()

if len(q2_all_stats_df) > 0:
    q2_all_stats_df.to_csv(Q2_STATS_CSV, index=False)
    save_json(
        {"rows": [to_serializable_row(r) for r in q2_all_stats_df.to_dict(orient="records")]},
        Q2_STATS_JSON
    )

if len(q2_all_cases_df) > 0:
    q2_all_cases_df.to_csv(Q2_CASES_CSV, index=False)
    save_json(
        {"rows": [to_serializable_row(r) for r in q2_all_cases_df.to_dict(orient="records")]},
        Q2_CASES_JSON
    )

if len(q2_all_top_df) > 0:
    q2_all_top_df.to_csv(Q2_TOP_CASES_CSV, index=False)

print("Saved Q2 all-pairs results:")
if len(q2_all_stats_df) > 0:
    print(" ", Q2_STATS_CSV)
    print(" ", Q2_STATS_JSON)
if len(q2_all_cases_df) > 0:
    print(" ", Q2_CASES_CSV)
    print(" ", Q2_CASES_JSON)
if len(q2_all_top_df) > 0:
    print(" ", Q2_TOP_CASES_CSV)

# %% [markdown]
# ### 3. How does attention agreement change across label frequency?
# 
# We are especially interested in whether agreement degrades for rare diagnoses compared to frequent ones.
# 
# We quantify label frequency (train prevalence by default), then test whether Q1 attention agreement
# (cos_mean / jac_mean) varies with frequency across *all* model pairs. Use train dataset because that frequency is what the model is based on which matters when looking at how frequency affects the models performace (causal). Use test as a sanity check as label distribution being wildly different at test time coulld require caution (evaluative). Exclude validation because that would just be noise as it was not used for training.
# 
# We report:
# - Spearman correlation (freq vs cosine/jaccard) per pair
# - Scatter plots per pair
# - Binned trends (rare → frequent) per pair

# %%
# Q3: label frequency (train + test)

def label_counts_from_loader(loader: DataLoader, n_labels: int = 50) -> tuple[np.ndarray, int]:
    counts = np.zeros(n_labels, dtype=int)
    total = 0
    for _, yb in loader:
        y_np = yb.detach().cpu().numpy()
        counts += y_np.sum(axis=0).astype(int)
        total += y_np.shape[0]
    return counts, total

train_counts, n_train = label_counts_from_loader(train_loader, n_labels=50)
test_counts,  n_test  = label_counts_from_loader(test_loader,  n_labels=50)

freq_df = pd.DataFrame({
    "label": np.arange(50),
    "train_pos": train_counts,
    "train_prev": train_counts / max(1, n_train),   #fraction of training sample where label is positive
    "test_pos": test_counts,
    "test_prev": test_counts / max(1, n_test),
}).sort_values("train_pos", ascending=False)

display(freq_df.head(10))
display(freq_df.tail(10))


# %%
# Q3: merge frequency into Q1 label×pair results (ALL pairs)

Q3_FREQ_COL = "train_pos"     # choose: "train_pos", "train_prev", "test_pos", "test_prev"
Q3_MIN_KEPT = 30              # drop label×pair rows with too few agree-positive samples

q3_df = (
    q1_df.merge(freq_df[["label", Q3_FREQ_COL]], on="label", how="left")
         .rename(columns={Q3_FREQ_COL: "label_freq"})
)

q3_df_filt = q3_df[q3_df["n_kept"] >= Q3_MIN_KEPT].copy()

print("Q3 rows (all pairs):", len(q3_df), " | after n_kept filter:", len(q3_df_filt))
display(q3_df_filt.head())


# %%
# Q3: focus view — Centralized vs Fed* only (recommended for reporting)

q3_central = q3_df_filt[q3_df_filt["pair"].str.contains("Centralized")].copy()
print("Centralized-vs-* rows:", len(q3_central))
display(q3_central.head())


# %%
# Q3: Spearman correlation per pair (ALL pairs)

def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    xr = xr - xr.mean()
    yr = yr - yr.mean()
    denom = (np.sqrt((xr**2).sum()) * np.sqrt((yr**2).sum())) + 1e-12
    return float((xr * yr).sum() / denom)

q3_corr_rows = []
for pair, sub in q3_df_filt.groupby("pair"):
    x = sub["label_freq"].to_numpy(dtype=float)
    q3_corr_rows.append({
        "pair": pair,
        "n_labels_used": int(sub["label"].nunique()),
        "total_kept": int(sub["n_kept"].sum()),
        "spearman_freq_vs_cos": spearman_corr(x, sub["cos_mean"].to_numpy(dtype=float)),
        "spearman_freq_vs_jac": spearman_corr(x, sub["jac_mean"].to_numpy(dtype=float)),
    })

q3_corr_df = pd.DataFrame(q3_corr_rows).sort_values(
    ["spearman_freq_vs_jac", "spearman_freq_vs_cos"], ascending=False
)

display(q3_corr_df)


# %%
# Q3: Scatter plots per pair (ALL pairs)
# Use log10(freq+1) because frequencies are heavy-tailed.

q3_plot = q3_df_filt.copy()
q3_plot["log_freq"] = np.log10(q3_plot["label_freq"] + 1)

for pair, sub in q3_plot.groupby("pair"):
    plt.figure()
    plt.scatter(sub["log_freq"], sub["cos_mean"])
    plt.title(f"Q3: Cosine vs Label Frequency | {pair}")
    plt.xlabel("log10(label frequency + 1)")
    plt.ylabel("Cosine mean (agree-positive)")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.scatter(sub["log_freq"], sub["jac_mean"])
    plt.title(f"Q3: Jaccard@{int(sub['top_k'].iloc[0])} vs Label Frequency | {pair}")
    plt.xlabel("log10(label frequency + 1)")
    plt.ylabel("Jaccard mean (agree-positive)")
    plt.tight_layout()
    plt.show()


# %%
# Q3: Binned trends (rare → frequent) per pair (ALL pairs)

Q3_NUM_BINS = 5  # quintiles

def binned_trend_all_pairs(df: pd.DataFrame, num_bins: int = 5) -> pd.DataFrame:
    out = []
    for pair, sub in df.groupby("pair"):
        # one row per label for this pair (already label×pair, but keep safe)
        tmp = sub[["label", "label_freq", "cos_mean", "jac_mean"]].drop_duplicates("label").copy()

        # bin labels by frequency rank within this pair
        tmp["bin"] = pd.qcut(tmp["label_freq"].rank(method="first"), q=num_bins, labels=False)

        g = tmp.groupby("bin").agg(
            labels_in_bin=("label", "count"),
            freq_min=("label_freq", "min"),
            freq_max=("label_freq", "max"),
            cos_mean=("cos_mean", "mean"),
            jac_mean=("jac_mean", "mean"),
        ).reset_index()

        g["pair"] = pair
        out.append(g)

    return pd.concat(out, ignore_index=True)

q3_bins = binned_trend_all_pairs(q3_df_filt, num_bins=Q3_NUM_BINS)
display(q3_bins.sort_values(["pair", "bin"]))

# Plot binned curves: one figure per metric, lines per pair
plt.figure()
for pair, sub in q3_bins.groupby("pair"):
    plt.plot(sub["bin"], sub["cos_mean"], marker="o", label=pair)
plt.title("Q3: Mean Cosine by Frequency Bin (rare → frequent) — ALL pairs")
plt.xlabel("Frequency bin (0=rarest)")
plt.ylabel("Mean cosine")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
for pair, sub in q3_bins.groupby("pair"):
    plt.plot(sub["bin"], sub["jac_mean"], marker="o", label=pair)
plt.title("Q3: Mean Jaccard by Frequency Bin (rare → frequent) — ALL pairs")
plt.xlabel("Frequency bin (0=rarest)")
plt.ylabel("Mean Jaccard")
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ### 4. Which federated methods behave closest to centralized training in terms of attention structure?
# 
# This allows us to compare federated methods beyond predictive performance and look at behavioral alignment.
# 
# We quantify "closeness" using Q1 agreement metrics between Centralized and each federated method:
# - weighted mean cosine similarity (higher = closer)
# - weighted mean Jaccard@K overlap (higher = closer)
# Weights = coverage (`n_kept`) per label.
# 
# We also report robustness:
# - per-label "win rate" (how often each method beats the others on a label)
# - bootstrap confidence intervals over labels (weighted)
# - optional breakdown by label frequency bins

# %%
# Q4: Extract Centralized-vs-Fed rows from Q1 and rank methods

FED_METHODS = ["FedAvg", "FedProx", "SCAFFOLD"]
CENTRAL = "Centralized"

# 1) Method-level summary (from q1_pair_summary if available)
if "q1_pair_summary" not in globals():
    raise RuntimeError("q1_pair_summary not found. Run Q1 aggregation cell first.")

q4_summary = q1_pair_summary[q1_pair_summary["pair"].str.contains(CENTRAL)].copy()
display(q4_summary)

# 2) Parse out the federated method name from "Centralized vs X"
def extract_fed_method(pair_str: str) -> str:
    # handles "Centralized vs FedAvg" etc.
    parts = pair_str.split(" vs ")
    return parts[1].strip() if len(parts) == 2 else pair_str

q4_summary["fed_method"] = q4_summary["pair"].apply(extract_fed_method)

# Keep only expected fed methods
q4_summary = q4_summary[q4_summary["fed_method"].isin(FED_METHODS)].copy()

# Rank by weighted metrics
q4_rank = q4_summary.sort_values(["jac_mean_weighted", "cos_mean_weighted"], ascending=False)[
    ["fed_method", "labels_covered", "total_kept", "cos_mean_weighted", "jac_mean_weighted",
     "cos_median_unweighted", "jac_median_unweighted"]
].reset_index(drop=True)

display(q4_rank)


# %%
# Q4: Bar plots (ranked) for Centralized vs Fed methods

if len(q4_rank) == 0:
    print("No Centralized-vs-Fed rows found. Check pair naming in q1_pair_summary.")
else:
    # cosine
    plt.figure()
    plt.bar(q4_rank["fed_method"], q4_rank["cos_mean_weighted"])
    plt.title("Q4: Closeness to Centralized (Weighted Cosine)")
    plt.ylabel("Weighted cosine similarity")
    plt.tight_layout()
    plt.show()

    # jaccard
    plt.figure()
    plt.bar(q4_rank["fed_method"], q4_rank["jac_mean_weighted"])
    plt.title(f"Q4: Closeness to Centralized (Weighted Jaccard@{int(q1_df['top_k'].iloc[0])})")
    plt.ylabel("Weighted Jaccard")
    plt.tight_layout()
    plt.show()


# %%
# Q4: Per-label "win rate" among Fed methods (which is closest to Centralized on each label?)

# Build a label×method table of Centralized-vs-method agreement
q4_label = q1_df[q1_df["pair"].str.contains(CENTRAL)].copy()
q4_label["fed_method"] = q4_label["pair"].apply(extract_fed_method)
q4_label = q4_label[q4_label["fed_method"].isin(FED_METHODS)].copy()

# Optional: only consider labels where all three methods have enough coverage
Q4_MIN_KEPT_PER_LABEL = 30
pivot_cos = q4_label.pivot_table(index="label", columns="fed_method", values="cos_mean")
pivot_jac = q4_label.pivot_table(index="label", columns="fed_method", values="jac_mean")
pivot_kept = q4_label.pivot_table(index="label", columns="fed_method", values="n_kept")

valid_labels = pivot_kept.dropna().index[
    (pivot_kept.dropna() >= Q4_MIN_KEPT_PER_LABEL).all(axis=1)
]

pivot_cos_f = pivot_cos.loc[valid_labels]
pivot_jac_f = pivot_jac.loc[valid_labels]

print("Labels where ALL 3 methods have n_kept >= threshold:", len(valid_labels), "/ 50")

# Winner per label
cos_winner = pivot_cos_f.idxmax(axis=1)
jac_winner = pivot_jac_f.idxmax(axis=1)

win_df = pd.DataFrame({
    "cos_winner": cos_winner.value_counts(),
    "jac_winner": jac_winner.value_counts(),
}).fillna(0).astype(int)

display(win_df)

# Also show some "disagreement labels" where cosine and jaccard winners differ
diff_winner_labels = cos_winner.index[cos_winner != jac_winner]
print("Labels where cosine-winner != jaccard-winner:", len(diff_winner_labels))
display(pd.DataFrame({
    "label": diff_winner_labels,
    "cos_winner": cos_winner.loc[diff_winner_labels].values,
    "jac_winner": jac_winner.loc[diff_winner_labels].values,
}).head(15))


# %%
# Q4: Bootstrap confidence intervals over labels (weighted by n_kept)

rng = np.random.default_rng(42)

def bootstrap_weighted_mean(sub_df: pd.DataFrame, value_col: str, weight_col: str, n_boot: int = 2000) -> tuple[float,float,float]:
    """
    Bootstrap over labels: resample labels with replacement, and compute weighted mean within each sample.
    """
    # one row per label
    d = sub_df[["label", value_col, weight_col]].dropna().copy()
    labels = d["label"].to_numpy()
    vals = d[value_col].to_numpy(dtype=float)
    wts  = d[weight_col].to_numpy(dtype=float)

    # map label -> (val, wt)
    by_label = {}
    for lbl, v, w in zip(labels, vals, wts):
        by_label[int(lbl)] = (float(v), float(w))
    uniq = np.array(list(by_label.keys()), dtype=int)

    def wmean_for_labels(sample_labels: np.ndarray) -> float:
        vv = np.array([by_label[int(l)][0] for l in sample_labels], dtype=float)
        ww = np.array([by_label[int(l)][1] for l in sample_labels], dtype=float)
        ww = np.clip(ww, 0.0, None)
        if ww.sum() <= 0:
            return float("nan")
        return float((vv * ww).sum() / ww.sum())

    boot = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        boot.append(wmean_for_labels(samp))

    boot = np.array(boot, dtype=float)
    center = float(np.nanmean(boot))
    lo = float(np.nanpercentile(boot, 2.5))
    hi = float(np.nanpercentile(boot, 97.5))
    return center, lo, hi

q4_ci_rows = []
for m in FED_METHODS:
    sub = q4_label[q4_label["fed_method"] == m].copy()
    # use n_kept as weights
    cos_c, cos_lo, cos_hi = bootstrap_weighted_mean(sub, "cos_mean", "n_kept", n_boot=2000)
    jac_c, jac_lo, jac_hi = bootstrap_weighted_mean(sub, "jac_mean", "n_kept", n_boot=2000)
    q4_ci_rows.append({
        "fed_method": m,
        "cos_mean_weighted_boot": cos_c,
        "cos_95ci": (cos_lo, cos_hi),
        "jac_mean_weighted_boot": jac_c,
        "jac_95ci": (jac_lo, jac_hi),
    })

q4_ci = pd.DataFrame(q4_ci_rows).sort_values(["jac_mean_weighted_boot","cos_mean_weighted_boot"], ascending=False)
display(q4_ci)


# %%
# Q4 (optional): Breakdown by label frequency bins (uses freq_df from Q3)

if "freq_df" not in globals():
    print("freq_df not found (run Q3 frequency cell first) — skipping frequency-bin breakdown.")
else:
    Q4_FREQ_COL = "train_pos"
    bins = 5

    tmp = q4_label.merge(freq_df[["label", Q4_FREQ_COL]], on="label", how="left").rename(columns={Q4_FREQ_COL: "label_freq"})
    # bin labels globally by frequency
    tmp_labels = tmp[["label","label_freq"]].drop_duplicates().copy()
    tmp_labels["bin"] = pd.qcut(tmp_labels["label_freq"].rank(method="first"), q=bins, labels=False)
    tmp = tmp.merge(tmp_labels[["label","bin"]], on="label", how="left")

    # weighted means within each (method, bin)
    out = []
    for (m, b), sub in tmp.groupby(["fed_method","bin"]):
        w = sub["n_kept"].to_numpy(dtype=float)
        out.append({
            "fed_method": m,
            "bin": int(b),
            "labels_in_bin": int(sub["label"].nunique()),
            "total_kept": int(sub["n_kept"].sum()),
            "cos_wmean": weighted_mean(sub["cos_mean"].to_numpy(dtype=float), w),
            "jac_wmean": weighted_mean(sub["jac_mean"].to_numpy(dtype=float), w),
        })

    q4_bins = pd.DataFrame(out).sort_values(["bin","fed_method"])
    display(q4_bins)

    # plot trends
    plt.figure()
    for m, sub in q4_bins.groupby("fed_method"):
        plt.plot(sub["bin"], sub["cos_wmean"], marker="o", label=m)
    plt.title("Q4: Centralized-vs-Fed closeness by label-frequency bin (Cosine)")
    plt.xlabel("Frequency bin (0=rarest)")
    plt.ylabel("Weighted cosine")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    for m, sub in q4_bins.groupby("fed_method"):
        plt.plot(sub["bin"], sub["jac_wmean"], marker="o", label=m)
    plt.title("Q4: Centralized-vs-Fed closeness by label-frequency bin (Jaccard)")
    plt.xlabel("Frequency bin (0=rarest)")
    plt.ylabel("Weighted Jaccard")
    plt.legend()
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ### 5. Finding if high agreement has more similar heatmaps and relating agreement to mislabeling
# 
# Highest-agreement samples (agree-positive) + Agreement ↔ Mislabeling (GT) analysis
# This section mirrors Q2, but:
# 1) Collects **ALL agree-positive rows** (no top-N-per-label) for unbiased agreement↔mislabeling analysis.
# 2) Also provides a **highest-agreement sample picker** for qualitative heatmap inspection per pair.
# 

# %%
# Q2H CONFIG

MODEL_NAMES = list(models.keys())

Q2H_PAIRS = [
    (a, b)
    for i, a in enumerate(MODEL_NAMES)
    for b in MODEL_NAMES[i+1:]
]

Q2H_LABELS = list(range(50))
Q2H_TOP_K = 15
Q2H_TRIM_CONTEXT = True
Q2H_CANDIDATE_POOL = "all_test"   # "all_test" or "gt_positive"
Q2H_MAX_SCAN_PER_LABEL = None     # speed knob

# qualitative inspection knobs
Q2H_TOP_OVERALL_N = 30            # show/keep top-N highest agreement cases overall per pair
Q2H_SHOW_QUAL = True
Q2H_N_QUAL_PER_PAIR = 3

# analysis knobs
Q2H_BINS = 10                     # number of quantile bins for agreement vs mislabeling

print("Q2H_LABELS:", (Q2H_LABELS[:10], "...") if len(Q2H_LABELS) > 10 else Q2H_LABELS)
print("Q2H_PAIRS:", Q2H_PAIRS)
print("Q2H_CANDIDATE_POOL:", Q2H_CANDIDATE_POOL)


# %%
# Helper: candidates (same logic as Q2 but using Q2H knobs)

def q2h_candidate_indices_for_label(label_idx: int) -> List[int]:
    if Q2H_CANDIDATE_POOL == "all_test":
        idxs = list(range(len(test_dataset)))
        return idxs[:Q2H_MAX_SCAN_PER_LABEL] if Q2H_MAX_SCAN_PER_LABEL else idxs

    if Q2H_CANDIDATE_POOL == "gt_positive":
        idxs = []
        for i in range(len(test_dataset)):
            _, y = test_dataset[i]
            if int(y[label_idx].item()) == 1:
                idxs.append(i)
                if Q2H_MAX_SCAN_PER_LABEL and len(idxs) >= Q2H_MAX_SCAN_PER_LABEL:
                    break
        return idxs

    raise ValueError(f"Unknown Q2H_CANDIDATE_POOL={Q2H_CANDIDATE_POOL}")


# %%
# Q2H: Collect ALL agree-positive rows for a given label + pair (NO per-label top-N)

@torch.no_grad()
def q2h_collect_agree_pos_rows_for_label(
    label_idx: int,
    model_a: str,
    model_b: str,
    top_k: int,
    trim_context: bool,
    candidate_indices: List[int],
) -> pd.DataFrame:
    rows = []
    kept = 0

    for ds_idx in candidate_indices:
        x, y = test_dataset[ds_idx]
        Xb = x.unsqueeze(0)

        # agree-positive filter (same prediction == positive)
        if not bool(agree_positive_mask(model_a, model_b, Xb, label_idx).item()):
            continue

        kept += 1

        window_size = config["window_size"] if trim_context else None
        valid_pos = get_valid_positions(x, PAD_INDEX, window_size)

        # attention vectors
        _, att_a = get_label_logits_and_attn(models[model_a], Xb, label_idx)
        _, att_b = get_label_logits_and_attn(models[model_b], Xb, label_idx)
        att_a_1d, att_b_1d = att_a[0], att_b[0]

        cos = cosine_sim_on_valid(att_a_1d, att_b_1d, valid_pos)
        jac = jaccard(
            topk_positions(att_a_1d, valid_pos, top_k),
            topk_positions(att_b_1d, valid_pos, top_k),
        )

        # prediction context
        logit_a, prob_a, pred_a = predict_label_for_samples(model_a, Xb, label_idx)
        logit_b, prob_b, pred_b = predict_label_for_samples(model_b, Xb, label_idx)
        thr_a = float(per_label_thr_by_model[model_a][label_idx].item())
        thr_b = float(per_label_thr_by_model[model_b][label_idx].item())

        rows.append({
            "label": label_idx,
            "sample": ds_idx,
            "pair": f"{model_a} vs {model_b}",
            "gt": int(y[label_idx].item()),

            "cos": float(cos),
            "jac": float(jac),

            "prob_a": float(prob_a[0].item()),
            "thr_a": thr_a,
            "margin_a": float(prob_a[0].item()) - thr_a,

            "prob_b": float(prob_b[0].item()),
            "thr_b": thr_b,
            "margin_b": float(prob_b[0].item()) - thr_b,

            "valid_len": int(len(valid_pos)),
        })

    df = pd.DataFrame(rows)
    df.attrs["n_kept"] = kept
    df.attrs["n_scanned"] = len(candidate_indices)
    return df


# %%
# Q2H: Run for a single pair -> returns:
#   1) label_stats_df (coverage)
#   2) agree_pos_df   (ALL agree-positive rows across labels, unbiased)

def q2h_run_for_pair_collect_all(model_a: str, model_b: str):
    all_rows = []
    label_stats = []

    for label_idx in tqdm(Q2H_LABELS, desc=f"Q2H collect ({model_a} vs {model_b})"):
        cand_idxs = q2h_candidate_indices_for_label(label_idx)

        df = q2h_collect_agree_pos_rows_for_label(
            label_idx=label_idx,
            model_a=model_a,
            model_b=model_b,
            top_k=Q2H_TOP_K,
            trim_context=Q2H_TRIM_CONTEXT,
            candidate_indices=cand_idxs,
        )

        n_kept = df.attrs.get("n_kept", 0)
        label_stats.append({
            "pair": f"{model_a} vs {model_b}",
            "label": label_idx,
            "n_scanned": df.attrs.get("n_scanned", len(cand_idxs)),
            "n_agree_pos": n_kept,
            "cos_max": float(df["cos"].max()) if len(df) else np.nan,
            "jac_max": float(df["jac"].max()) if len(df) else np.nan,
        })

        if len(df) > 0:
            all_rows.append(df)

    label_stats_df = (
        pd.DataFrame(label_stats)
          .sort_values(["n_agree_pos", "cos_max"], ascending=[False, False])
          .reset_index(drop=True)
    )

    agree_pos_df = pd.concat(all_rows, ignore_index=True) if len(all_rows) else pd.DataFrame()

    # derived columns for analysis
    if len(agree_pos_df) > 0:
        agree_pos_df["is_fp"] = (agree_pos_df["gt"] == 0).astype(int)  # since agree-positive => GT=0 means shared FP
        agree_pos_df["agreement_avg"] = (agree_pos_df["cos"] + agree_pos_df["jac"]) / 2.0
        agree_pos_df["agreement_min"] = agree_pos_df[["cos", "jac"]].min(axis=1)

    return label_stats_df, agree_pos_df


# %%
# Q2H: Pick examples from highest-agreement table for qualitative inspection

def q2h_pick_examples_high_agree(df_top: pd.DataFrame, n: int = 3) -> list[pd.Series]:
    """
    Picks:
      1) highest agreement (rank 0)
      2) another high agreement (rank ~10 or last)
      3) different label if possible
    """
    if df_top is None or len(df_top) == 0:
        return []

    picks = []
    picks.append(df_top.iloc[0])

    mid_idx = min(10, len(df_top) - 1)
    if mid_idx != 0 and len(picks) < n:
        picks.append(df_top.iloc[mid_idx])

    used_labels = {int(r["label"]) for r in picks}
    if len(picks) < n:
        for _, row in df_top.iterrows():
            if int(row["label"]) not in used_labels:
                picks.append(row)
                break

    return picks[:n]


# %%
# Q2H: Agreement ↔ mislabeling analysis (per pair)
# - Works on ALL agree-positive rows (unbiased).
# - Produces:
#   * overall FP rate
#   * FP rate by agreement quantile bins for cos, jac, and combined scores

def q2h_agreement_vs_mislabeling(agree_pos_df: pd.DataFrame, n_bins: int = 10) -> dict:
    if agree_pos_df is None or len(agree_pos_df) == 0:
        return {"summary": None, "by_bin": {}}

    out = {}

    # overall summary
    summary = {
        "n_rows": int(len(agree_pos_df)),
        "fp_rate": float(agree_pos_df["is_fp"].mean()),
        "tp_rate": float((agree_pos_df["gt"] == 1).mean()),
        "cos_mean": float(agree_pos_df["cos"].mean()),
        "jac_mean": float(agree_pos_df["jac"].mean()),
        "agreement_avg_mean": float(agree_pos_df["agreement_avg"].mean()),
        "agreement_min_mean": float(agree_pos_df["agreement_min"].mean()),
    }
    out["summary"] = pd.DataFrame([summary])

    by_bin = {}

    def _bin_stats(col: str) -> pd.DataFrame:
        df = agree_pos_df.copy()

        # qcut can fail if too many duplicate values; handle with rank-based fallback
        try:
            df["bin"] = pd.qcut(df[col], q=n_bins, duplicates="drop")
        except ValueError:
            df["bin"] = pd.qcut(df[col].rank(method="average"), q=n_bins, duplicates="drop")

        g = df.groupby("bin", observed=True).agg(
            n=("is_fp", "size"),
            fp_rate=("is_fp", "mean"),
            cos_mean=("cos", "mean"),
            jac_mean=("jac", "mean"),
            agreement_avg_mean=("agreement_avg", "mean"),
            agreement_min_mean=("agreement_min", "mean"),
            gt_pos_rate=("gt", "mean"),
        ).reset_index()

        # add bin endpoints for readability
        # (bin is an interval in most cases)
        return g.sort_values("bin")

    for col in ["cos", "jac", "agreement_avg", "agreement_min"]:
        by_bin[col] = _bin_stats(col)

    out["by_bin"] = by_bin
    return out


# %%
# Q2H: Run ALL pairs
# For each pair:
#   1) collect ALL agree-positive rows (agree_pos_df)
#   2) show high-agreement examples (top overall) for inspection
#   3) run agreement↔mislabeling analysis tables

q2h_pair_results = {}  # (a,b) -> dict

for a, b in Q2H_PAIRS:
    label_stats_df, agree_pos_df = q2h_run_for_pair_collect_all(a, b)

    print("\n==============================")
    print(f"PAIR: {a} vs {b}")
    print("Label coverage (top 10 by agree-positive):")
    display(label_stats_df.head(10))

    if agree_pos_df is None or len(agree_pos_df) == 0:
        print("No agree-positive rows found for this pair.")
        q2h_pair_results[(a, b)] = {
            "label_stats": label_stats_df,
            "agree_pos": agree_pos_df,
            "top_high_agree": pd.DataFrame(),
            "analysis": {"summary": None, "by_bin": {}},
        }
        continue

    # Highest-agreement rows overall (global, across labels)
    top_high_agree = (
        agree_pos_df.sort_values(["cos", "jac"], ascending=[False, False])
                    .head(Q2H_TOP_OVERALL_N)
                    .reset_index(drop=True)
    )

    print(f"Highest-agreement cases overall (top {Q2H_TOP_OVERALL_N}):")
    display(top_high_agree)

    # Agreement vs mislabeling analysis (unbiased)
    analysis = q2h_agreement_vs_mislabeling(agree_pos_df, n_bins=Q2H_BINS)

    print("Agreement↔mislabeling summary:")
    display(analysis["summary"])

    print("FP rate by COS agreement quantiles:")
    display(analysis["by_bin"]["cos"])

    print("FP rate by JACCARD agreement quantiles:")
    display(analysis["by_bin"]["jac"])

    print("FP rate by AVG agreement quantiles ( (cos+jac)/2 ):")
    display(analysis["by_bin"]["agreement_avg"])

    print("FP rate by MIN agreement quantiles ( min(cos,jac) ):")
    display(analysis["by_bin"]["agreement_min"])

    # Qualitative inspection: inspect a few high-agreement examples
    if Q2H_SHOW_QUAL and len(top_high_agree) > 0:
        examples = q2h_pick_examples_high_agree(top_high_agree, n=Q2H_N_QUAL_PER_PAIR)
        for ex_row in examples:
            q2_inspect_case(
                ex_row,
                top_k=Q2H_TOP_K,
                trim_context=Q2H_TRIM_CONTEXT,
                do_position_heatmap=True
            )

    q2h_pair_results[(a, b)] = {
        "label_stats": label_stats_df,
        "agree_pos": agree_pos_df,
        "top_high_agree": top_high_agree,
        "analysis": analysis,
    }


# %%
# Combine all pairs' agree-positive rows into one big df (tagged by pair)

all_pairs_agree_pos = []
for (a, b), d in q2h_pair_results.items():
    df = d.get("agree_pos", None)
    if df is not None and len(df) > 0:
        all_pairs_agree_pos.append(df)

all_pairs_agree_pos_df = pd.concat(all_pairs_agree_pos, ignore_index=True) if len(all_pairs_agree_pos) else pd.DataFrame()
print("Total agree-positive rows across all pairs:", len(all_pairs_agree_pos_df))
display(all_pairs_agree_pos_df.head(10))

if len(all_pairs_agree_pos_df) > 0:
    agg_analysis = q2h_agreement_vs_mislabeling(all_pairs_agree_pos_df, n_bins=Q2H_BINS)
    print("AGGREGATE summary (all pairs):")
    display(agg_analysis["summary"])

    print("AGGREGATE FP rate by COS agreement quantiles:")
    display(agg_analysis["by_bin"]["cos"])

    print("AGGREGATE FP rate by JACCARD agreement quantiles:")
    display(agg_analysis["by_bin"]["jac"])


# %%
def plot_fp_rate_by_bin(bin_df: pd.DataFrame, title: str):
    if bin_df is None or len(bin_df) == 0:
        print("No data to plot.")
        return
    # Use bin order on x; matplotlib can handle categorical via range
    x = list(range(len(bin_df)))
    y = bin_df["fp_rate"].values

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xticks(x, [str(b) for b in bin_df["bin"]], rotation=45, ha="right")
    plt.ylabel("False Positive Rate (GT=0 among agree-positive)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Example: plot aggregate if present
if "agg_analysis" in globals() and agg_analysis["by_bin"]:
    plot_fp_rate_by_bin(agg_analysis["by_bin"]["cos"], "Aggregate: FP rate vs COS agreement quantiles")
    plot_fp_rate_by_bin(agg_analysis["by_bin"]["jac"], "Aggregate: FP rate vs JACCARD agreement quantiles")
    plot_fp_rate_by_bin(agg_analysis["by_bin"]["agreement_avg"], "Aggregate: FP rate vs AVG agreement quantiles")
    plot_fp_rate_by_bin(agg_analysis["by_bin"]["agreement_min"], "Aggregate: FP rate vs MIN agreement quantiles")


# %% [markdown]
# ## Phrase Level Attention Inspection

# %%
# Phrase / window extraction helpers with merging

def _merge_overlapping_spans(spans):
    """spans: list[(start,end,score)] merges overlaps keeping max score (as before)."""
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    merged = [list(spans[0])]
    for s, e, sc in spans[1:]:
        ms, me, msc = merged[-1]
        if s <= me:  # overlap
            merged[-1][1] = max(me, e)
            merged[-1][2] = max(msc, sc)
        else:
            merged.append([s, e, sc])
    return [tuple(x) for x in merged]


def top_attention_phrases_merging(
    tokens,
    attn_1d,
    valid_pos=None,
    top_k=5,
    window_size=10,
    centered=True,
    skip_pad=True,
    return_attn=True,
    merge_overlaps=True,   # NEW: toggle merging
):
    """
    Returns list of (start, end, score, text, attn_vals, span_tokens).
    score = sum(attn over span), like the Colab output.

    Notes:
      - Windows are built around top-attended token positions.
      - If merge_overlaps=True, overlapping windows are merged into larger spans.
      - If merge_overlaps=False, returned spans are the best scored unique windows after dedup.
    """
    L = len(tokens)
    att = attn_1d.detach().cpu().numpy()

    if valid_pos is None:
        cand_pos = np.arange(L)
    else:
        cand_pos = np.array(valid_pos, dtype=int)

    if skip_pad:
        cand_pos = np.array([p for p in cand_pos if tokens[p] != "<PAD>"], dtype=int)
        if cand_pos.size == 0:
            return []

    att_c = att[cand_pos]
    grab = min(len(cand_pos), top_k * 5)
    top_idx = np.argsort(att_c)[::-1][:grab]
    top_positions = cand_pos[top_idx]

    spans = []
    half = window_size // 2

    for pos in top_positions:
        if centered:
            start = max(int(pos) - half, 0)
            end   = min(start + window_size, L)
            start = max(end - window_size, 0)
        else:
            start = int(pos)
            end   = min(start + window_size, L)

        # raw tokens + raw attention in the span (BEFORE any PAD filtering)
        span_tokens_raw = tokens[start:end]
        span_att_raw = att[start:end]

        # score matches Colab: sum of attention in the window/span
        score = float(span_att_raw.sum())

        # optionally remove PAD tokens for display (keep aligned tokens+attn)
        if skip_pad:
            kept = [(t, a) for t, a in zip(span_tokens_raw, span_att_raw) if t != "<PAD>"]
            if not kept:
                continue
            span_tokens_disp = [t for t, _ in kept]
            span_att_disp = [float(a) for _, a in kept]
        else:
            span_tokens_disp = span_tokens_raw
            span_att_disp = [float(a) for a in span_att_raw]

        span_text = " ".join(span_tokens_disp).strip()
        if span_text:
            if return_attn:
                spans.append((start, end, score, span_text, span_att_disp, span_tokens_disp))
            else:
                spans.append((start, end, score, span_text))

    # dedup by text, keep best score
    best = {}
    for item in spans:
        txt = item[3]
        sc = item[2]
        if (txt not in best) or (sc > best[txt][2]):
            best[txt] = item
    spans = list(best.values())

    if not spans:
        return []

    # NEW: allow skipping merge. If skipping, just sort by score and return top_k.
    if not merge_overlaps:
        spans.sort(key=lambda x: -x[2])
        return spans[:top_k]

    # merge overlaps (keeping max score span)
    spans_simple = [(s, e, sc) for (s, e, sc, *_rest) in spans]
    merged = _merge_overlapping_spans(spans_simple)

    # rebuild merged spans with text + (optional) attn vector
    out = []
    for s, e, sc in merged:
        span_tokens_raw = tokens[s:e]
        span_att_raw = att[s:e]
        if skip_pad:
            kept = [(t, a) for t, a in zip(span_tokens_raw, span_att_raw) if t != "<PAD>"]
            if not kept:
                continue
            span_tokens_disp = [t for t, _ in kept]
            span_att_disp = [float(a) for _, a in kept]
        else:
            span_tokens_disp = span_tokens_raw
            span_att_disp = [float(a) for a in span_att_raw]

        txt = " ".join(span_tokens_disp).strip()
        if txt:
            sc2 = float(np.sum(span_att_raw))
            if return_attn:
                out.append((s, e, sc2, txt, span_att_disp, span_tokens_disp))
            else:
                out.append((s, e, sc2, txt))

    out.sort(key=lambda x: -x[2])
    return out[:top_k]

# %%
# Phrase / window extraction helpers (with span-level NMS)

def span_iou(a, b):
    """IoU for half-open token spans a=(s1,e1), b=(s2,e2)."""
    s1, e1 = a
    s2, e2 = b
    inter = max(0, min(e1, e2) - max(s1, s2))
    if inter <= 0:
        return 0.0
    union = (e1 - s1) + (e2 - s2) - inter
    return float(inter) / float(union) if union > 0 else 0.0


def nms_spans(spans, top_k=5, iou_thresh=0.5):
    """
    Non-maximum suppression over spans.

    spans: list of tuples that start with (start, end, score, ...)
           e.g. (start, end, score, text, attn_vals, span_tokens)

    Keeps highest-score spans, suppresses any candidate whose IoU with any kept span >= iou_thresh.
    """
    if not spans:
        return []

    spans_sorted = sorted(spans, key=lambda x: -x[2])  # sort by score desc
    kept = []

    for item in spans_sorted:
        s, e = item[0], item[1]
        too_close = any(span_iou((s, e), (ks, ke)) >= iou_thresh for (ks, ke, *_rest) in kept)
        if too_close:
            continue
        kept.append(item)
        if len(kept) >= top_k:
            break

    return kept


def top_attention_phrases_NMS(
    tokens,
    attn_1d,
    valid_pos=None,
    top_k=5,
    window_size=10,
    centered=True,
    skip_pad=True,
    return_attn=True,
    # NEW: NMS controls
    use_nms=True,
    nms_iou_thresh=0.3,
    nms_grab_multiplier=5,   # grab more candidates before NMS
):
    """
    Returns list of (start, end, score, text, attn_vals, span_tokens).
    score = sum(attn over span) on the RAW window [start:end].

    This version uses fixed-length windows and (optionally) NMS to ensure non-overlapping phrases.
    - Windows are built around top-attended token positions.
    - use_nms=True keeps spans fixed-length and enforces distinctness via IoU suppression.
    """
    L = len(tokens)
    att = attn_1d.detach().cpu().numpy()

    if valid_pos is None:
        cand_pos = np.arange(L)
    else:
        cand_pos = np.array(valid_pos, dtype=int)

    if skip_pad:
        cand_pos = np.array([p for p in cand_pos if tokens[p] != "<PAD>"], dtype=int)
        if cand_pos.size == 0:
            return []

    # pick top-attended token positions (grab more than top_k so NMS has options)
    att_c = att[cand_pos]
    grab = min(len(cand_pos), max(top_k * nms_grab_multiplier, top_k))
    top_idx = np.argsort(att_c)[::-1][:grab]
    top_positions = cand_pos[top_idx]

    spans = []
    half = window_size // 2

    for pos in top_positions:
        if centered:
            start = max(int(pos) - half, 0)
            end   = min(start + window_size, L)
            start = max(end - window_size, 0)
        else:
            start = int(pos)
            end   = min(start + window_size, L)

        # raw tokens + raw attention in the fixed window (BEFORE any PAD filtering)
        span_tokens_raw = tokens[start:end]
        span_att_raw = att[start:end]

        # score matches Colab: sum of attention in the window/span
        score = float(span_att_raw.sum())

        # optionally remove PAD tokens for display (keep aligned tokens+attn)
        if skip_pad:
            kept_tok_att = [(t, a) for t, a in zip(span_tokens_raw, span_att_raw) if t != "<PAD>"]
            if not kept_tok_att:
                continue
            span_tokens_disp = [t for t, _ in kept_tok_att]
            span_att_disp = [float(a) for _, a in kept_tok_att]
        else:
            span_tokens_disp = span_tokens_raw
            span_att_disp = [float(a) for a in span_att_raw]

        span_text = " ".join(span_tokens_disp).strip()
        if not span_text:
            continue

        if return_attn:
            spans.append((start, end, score, span_text, span_att_disp, span_tokens_disp))
        else:
            spans.append((start, end, score, span_text))

    if not spans:
        return []

    # dedup by text, keep best score (still useful even with NMS)
    best = {}
    for item in spans:
        txt = item[3]
        sc = item[2]
        if (txt not in best) or (sc > best[txt][2]):
            best[txt] = item
    spans = list(best.values())

    if not spans:
        return []

    # apply NMS to enforce distinct spans while keeping fixed window sizes
    if use_nms:
        spans = nms_spans(spans, top_k=top_k, iou_thresh=nms_iou_thresh)
        spans.sort(key=lambda x: -x[2])
        return spans[:top_k]

    # fallback: no NMS, just top by score (can be highly redundant)
    spans.sort(key=lambda x: -x[2])
    return spans[:top_k]


# %%
# # NMS threshold sweep: find a good iou / overlap threshold for phrase extraction

# # -- helper: overlap ratio defined as intersection / min(len(A), len(B))
# def span_overlap_ratio_minlen(a, b):
#     s1, e1 = a
#     s2, e2 = b
#     inter = max(0, min(e1, e2) - max(s1, s2))
#     if inter <= 0:
#         return 0.0
#     minlen = min(e1 - s1, e2 - s2)
#     return float(inter) / float(minlen) if minlen > 0 else 0.0

# # -- helper: pairwise IoU summary for a list of spans
# def pairwise_overlap_stats(spans, metric="iou"):
#     # spans: list of tuples (s,e,score,...)
#     if len(spans) < 2:
#         return {"mean_pairwise": 0.0, "max_pairwise": 0.0}
#     vals = []
#     for i, j in combinations(range(len(spans)), 2):
#         a = (spans[i][0], spans[i][1])
#         b = (spans[j][0], spans[j][1])
#         if metric == "iou":
#             vals.append(span_iou(a, b))
#         elif metric == "minlen":
#             vals.append(span_overlap_ratio_minlen(a, b))
#         else:
#             raise ValueError("metric must be 'iou' or 'minlen'")
#     return {"mean_pairwise": float(np.mean(vals)), "max_pairwise": float(np.max(vals))}

# # -- sweep function
# @torch.no_grad()
# def sweep_nms_thresholds(
#     sample_size=300,
#     label_sample=None,              # list of label indices to evaluate (None -> random labels per sample)
#     model_names=("Centralized","FedAvg","FedProx","SCAFFOLD"),
#     thresholds=(0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5),
#     top_k=5,
#     phrase_window=12,
#     trim_context=True,
#     nms_grab_multiplier=5,
#     overlap_metric="iou",           # "iou" or "minlen"
#     random_seed=42,
# ):
#     """
#     Sweep NMS thresholds and return a DataFrame of aggregated statistics.

#     Returns:
#       df: pandas.DataFrame with rows (threshold, model) and columns:
#          mean_num_spans, std_num_spans, mean_mean_pairwise, mean_max_pairwise
#     """
#     rng = np.random.RandomState(random_seed)

#     # select sample indices (avoid too-large runs)
#     all_indices = list(range(len(test_dataset)))
#     if sample_size >= len(all_indices):
#         sample_indices = all_indices
#     else:
#         sample_indices = rng.choice(all_indices, size=sample_size, replace=False).tolist()

#     # if label_sample is None, we will randomly pick one *positive* label per sample if possible,
#     # otherwise we will evaluate all labels (or a fixed set). To keep runtime reasonable, we'll
#     # sample one label per sample unless label_sample is provided as a list.
#     # Here we choose a random label index in range(num_labels) as a default fallback.
#     num_labels = test_dataset[0][1].shape[0]  # assumes (x,y) and y is vector
#     if label_sample is None:
#         pick_label_per_sample = True
#     else:
#         pick_label_per_sample = False
#         label_sample = list(label_sample)

#     rows = []

#     for t in tqdm(thresholds, desc="thresholds"):
#         stats_by_model = {m: {"nspans": [], "mean_pairwise": [], "max_pairwise": []} for m in model_names}

#         for ds_idx in sample_indices:
#             x, y = test_dataset[ds_idx]
#             tokens = decode_sequence(x)
#             window_size = config["window_size"] if trim_context else None
#             valid_pos = get_valid_positions(x, PAD_INDEX, window_size)

#             # choose which labels to test: either provided list or one random label per sample
#             if pick_label_per_sample:
#                 lbl = rng.randint(0, num_labels)
#                 labels_to_check = [lbl]
#             else:
#                 labels_to_check = label_sample

#             for lbl in labels_to_check:
#                 for m in model_names:
#                     # get attention vector
#                     logits_lbl, attn_lbl = get_label_logits_and_attn(models[m], x.unsqueeze(0), lbl)
#                     attn_vec = attn_lbl[0]

#                     spans = top_attention_phrases_NMS(
#                         tokens=tokens,
#                         attn_1d=attn_vec,
#                         valid_pos=valid_pos,
#                         top_k=top_k,
#                         window_size=phrase_window,
#                         centered=True,
#                         skip_pad=True,
#                         return_attn=True,
#                         use_nms=True,
#                         nms_iou_thresh=float(t),
#                         nms_grab_multiplier=nms_grab_multiplier,
#                     )

#                     # record counts
#                     stats_by_model[m]["nspans"].append(len(spans))

#                     # pairwise overlap stats
#                     pair_stats = pairwise_overlap_stats(spans, metric=("iou" if overlap_metric=="iou" else "minlen"))
#                     stats_by_model[m]["mean_pairwise"].append(pair_stats["mean_pairwise"])
#                     stats_by_model[m]["max_pairwise"].append(pair_stats["max_pairwise"])

#         # aggregate per model
#         for m in model_names:
#             arr_n = np.array(stats_by_model[m]["nspans"])
#             arr_mp = np.array(stats_by_model[m]["mean_pairwise"])
#             arr_xp = np.array(stats_by_model[m]["max_pairwise"])
#             rows.append({
#                 "threshold": float(t),
#                 "model": m,
#                 "mean_num_spans": float(np.mean(arr_n)) if arr_n.size else 0.0,
#                 "std_num_spans": float(np.std(arr_n)) if arr_n.size else 0.0,
#                 "mean_mean_pairwise": float(np.mean(arr_mp)) if arr_mp.size else 0.0,
#                 "mean_max_pairwise": float(np.mean(arr_xp)) if arr_xp.size else 0.0,
#                 "sample_size": len(sample_indices),
#                 "top_k": top_k,
#                 "phrase_window": phrase_window,
#                 "overlap_metric": overlap_metric,
#             })

#     df = pd.DataFrame(rows)
#     return df

# # %%
# # Example usage (quick, medium, or longer runs)
# # - For a quick exploratory run, sample_size=200 and phrase_window=12 is reasonable.
# # - Increase sample_size for more stable estimates.
# df_results = sweep_nms_thresholds(
#     sample_size=250,
#     thresholds=(0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
#     top_k=5,
#     phrase_window=12,
#     nms_grab_multiplier=6,
#     overlap_metric="minlen",   # or "minlen"
# )

# # Show results pivoted
# display(df_results.pivot_table(index="threshold", columns="model",
#                                values=["mean_num_spans","mean_max_pairwise"]).round(4))

# # Save CSV for later inspection
# # df_results.to_csv("nms_threshold_sweep_results.csv", index=False)
# # print("Saved results to nms_threshold_sweep_results.csv")


# %%
# Compare across models display function (updated for NMS-based phrase extraction)

@torch.no_grad()
def show_phrase_attention_comparison(
    sample_idx: int,
    label_idx: int,
    model_names=("Centralized", "FedAvg", "FedProx", "SCAFFOLD"),
    top_k_phrases: int = 5,
    phrase_window: int = 10,
    trim_context: bool = True,
    # NEW: NMS controls
    use_nms: bool = True,
    nms_iou_thresh: float = 0.3,
    nms_grab_multiplier: int = 5,
):
    x, y = test_dataset[sample_idx]
    tokens = decode_sequence(x)

    window_size = config["window_size"] if trim_context else None
    valid_pos = get_valid_positions(x, PAD_INDEX, window_size)

    print(f"\n=== Phrase attention comparison | sample={sample_idx} label={label_idx} ({label_info(label_idx)}) ===")
    print(f"GT={int(y[label_idx].item())} | valid_len={len(valid_pos)} / seq_len={len(tokens)}")
    print(
        f"(phrase_window={phrase_window}, top_k_phrases={top_k_phrases}, trim_context={trim_context}, "
        f"use_nms={use_nms}, nms_iou_thresh={nms_iou_thresh})"
    )

    phrases_by_model = {}

    for mname in model_names:
        logits_lbl, attn_lbl = get_label_logits_and_attn(models[mname], x.unsqueeze(0), label_idx)
        prob = float(torch.sigmoid(logits_lbl[0]).item())
        thr  = float(per_label_thr_by_model[mname][label_idx].item())
        pred = bool(prob >= thr)

        attn_vec = attn_lbl[0]  # (L,)

        phrases = top_attention_phrases_NMS(
            tokens=tokens,
            attn_1d=attn_vec,
            valid_pos=valid_pos,
            top_k=top_k_phrases,
            window_size=phrase_window,
            centered=True,
            skip_pad=True,
            use_nms=use_nms,
            nms_iou_thresh=nms_iou_thresh,
            nms_grab_multiplier=nms_grab_multiplier,
        )
        phrases_by_model[mname] = phrases

        print(f"\n--- {mname} --- prob={prob:.4f} thr={thr:.2f} pred={pred}")
        for i, (s, e, score, txt, att_vals, span_toks) in enumerate(phrases, 1):
            print(f"#{i}  span[{s}:{e}]  score={score:.4f}  {txt}")
            print(f"     Tokens: {' '.join(span_toks)}")
            print(f"     Attention: {[round(a, 4) for a in att_vals]}")

    # simple overlap view: which phrase texts are shared? (exact-text match; optional sanity check)
    sets = {m: set([p[3] for p in ph]) for m, ph in phrases_by_model.items()}
    if len(sets) >= 2:
        base = model_names[0]
        print(f"\n=== Phrase overlap vs {base} (exact-text match) ===")
        base_set = sets.get(base, set())
        for m in model_names[1:]:
            inter = base_set & sets.get(m, set())
            print(f"{m}: {len(inter)} shared / {len(base_set)} centralized phrases")
            for t in list(inter)[:5]:
                print("  -", t)


# %%
# sample use
def find_agree_positive_example(label_idx, model_a="Centralized", model_b="FedAvg", max_scan=5000):
    for ds_idx in range(min(max_scan, len(test_dataset))):
        x, y = test_dataset[ds_idx]
        if bool(agree_positive_mask(model_a, model_b, x.unsqueeze(0), label_idx).item()):
            return ds_idx
    return None

lbl = 7
ds = find_agree_positive_example(lbl, "Centralized", "FedAvg")
print("found sample:", ds)

show_phrase_attention_comparison(
    sample_idx=ds,
    label_idx=lbl,
    model_names=("Centralized","FedAvg","FedProx","SCAFFOLD"),
    top_k_phrases=5,
    phrase_window=12,
    trim_context=True,
)

#specific test
x, y = test_dataset[1212]
print("GT:", int(y[1].item()))

logits, _ = get_label_logits_and_attn(models["Centralized"], x.unsqueeze(0), 1)
prob = torch.sigmoid(logits[0]).item()
thr  = per_label_thr_by_model["Centralized"][1].item()
print("Centralized prob/thr:", prob, thr)
sample_idx = 1212
label_idx  = 1
show_phrase_attention_comparison(sample_idx, label_idx)


# %%
# Phrase-level similarity across models
# Metrics: Top-1 IoU, Top-5 Max IoU, Weighted / Unweighted / Rank-aware Best-Match IoU (all symmetric)

# --- span overlap metric for comparing models ---
# Use IoU for comparison, independent of the NMS overlap criterion (minlen).
# (You can swap in span_overlap_ratio_minlen if you prefer.)
def span_iou_pair(a, b):
    return span_iou((a[0], a[1]), (b[0], b[1]))

def extract_phrases_for_model(x, label_idx, model_name, top_k=5, phrase_window=12,
                              trim_context=True, use_nms=True, nms_thresh=0.30, nms_grab_multiplier=6):
    tokens = decode_sequence(x)
    window_size = config["window_size"] if trim_context else None
    valid_pos = get_valid_positions(x, PAD_INDEX, window_size)

    logits_lbl, attn_lbl = get_label_logits_and_attn(models[model_name], x.unsqueeze(0), label_idx)
    attn_vec = attn_lbl[0]

    phrases = top_attention_phrases_NMS(
        tokens=tokens,
        attn_1d=attn_vec,
        valid_pos=valid_pos,
        top_k=top_k,
        window_size=phrase_window,
        centered=True,
        skip_pad=True,
        return_attn=True,
        use_nms=use_nms,
        nms_iou_thresh=float(nms_thresh),
        nms_grab_multiplier=int(nms_grab_multiplier),
    )
    # Each phrase: (start, end, score, text, attn_vals, span_tokens)
    return phrases

def top1_iou(A, B):
    if not A or not B:
        return np.nan
    return span_iou_pair(A[0], B[0])

def topk_max_iou(A, B):
    if not A or not B:
        return np.nan
    best = 0.0
    for a in A:
        for b in B:
            best = max(best, span_iou_pair(a, b))
    return best

def weighted_best_match_iou(A, B, eps=1e-12):
    """
    Primary metric:
      - For each phrase in A, find best IoU against any phrase in B.
      - Weighted average of these best IoUs using A's phrase scores and B's scores.
      - Symmetrize by averaging A->B and B->A.
    """
    def one_way(src, tgt):
        if not src or not tgt:
            return np.nan
        scores = np.array([max(span_iou_pair(s, t) for t in tgt) for s in src], dtype=float)
        w = np.array([max(s[2], 0.0) for s in src], dtype=float)
        if w.sum() <= eps:
            return float(np.mean(scores))  # fallback if all weights zero-ish
        return float((scores * w).sum() / (w.sum() + eps))

    ab = one_way(A, B)
    ba = one_way(B, A)
    if np.isnan(ab) and np.isnan(ba):
        return np.nan
    if np.isnan(ab):
        return ba
    if np.isnan(ba):
        return ab
    return 0.5 * (ab + ba)

def unweighted_best_match_iou(A, B):
    """
    Metric 4:
      - For each phrase in A, find best IoU against any phrase in B.
      - Average these best IoUs (no weights).
      - Symmetrize by averaging A->B and B->A.
    """
    def one_way(src, tgt):
        if not src or not tgt:
            return np.nan
        scores = np.array([max(span_iou_pair(s, t) for t in tgt) for s in src], dtype=float)
        return float(np.mean(scores)) if scores.size else np.nan

    ab = one_way(A, B)
    ba = one_way(B, A)
    if np.isnan(ab) and np.isnan(ba):
        return np.nan
    if np.isnan(ab):
        return ba
    if np.isnan(ba):
        return ab
    return 0.5 * (ab + ba)


def rank_weight(i, j, mode="inv", alpha=0.7):
    """
    Weight based on rank mismatch between phrase i in src and phrase j in tgt.
    Ranks are 0-indexed here, so use |i-j|.

    mode="inv": 1 / (1 + |i-j|)   (simple, bounded, interpretable)
    mode="exp": exp(-alpha * |i-j|) (stronger penalty if alpha>0)
    """
    d = abs(int(i) - int(j))
    if mode == "exp":
        return float(np.exp(-alpha * d))
    return float(1.0 / (1.0 + d))


def rankaware_best_match_iou(A, B, rank_mode="inv", alpha=0.7):
    """
    Metric 5:
      - For each phrase i in A, find the best match phrase j in B
        maximizing: IoU(i,j) * rank_weight(i,j)
      - Average over i (unweighted)
      - Symmetrize by averaging A->B and B->A

    This directly penalizes "1 matches 5" vs "1 matches 1".
    """
    def one_way(src, tgt):
        if not src or not tgt:
            return np.nan

        vals = []
        for i, s in enumerate(src):
            best = 0.0
            for j, t in enumerate(tgt):
                iou = span_iou_pair(s, t)
                w = rank_weight(i, j, mode=rank_mode, alpha=alpha)
                best = max(best, iou * w)
            vals.append(best)

        vals = np.array(vals, dtype=float)
        return float(np.mean(vals)) if vals.size else np.nan

    ab = one_way(A, B)
    ba = one_way(B, A)
    if np.isnan(ab) and np.isnan(ba):
        return np.nan
    if np.isnan(ab):
        return ba
    if np.isnan(ba):
        return ab
    return 0.5 * (ab + ba)

@torch.no_grad()
def run_phrase_similarity_eval(
    sample_indices,
    label_indices,
    model_names=("Centralized","FedAvg","FedProx","SCAFFOLD"),
    top_k=5,
    phrase_window=12,
    trim_context=True,
    nms_thresh=0.30,            # <-- use the tuned threshold here
    nms_grab_multiplier=6,
):
    rows = []

    # cache phrases so we don't recompute per pair
    for ds_idx in tqdm(sample_indices, desc="samples", disable=True):
        x, y = test_dataset[ds_idx]

        for lbl in label_indices:
            # optional: you can filter to positives if you want
            gt = int(y[lbl].item())

            phrases_by_model = {}
            for m in model_names:
                phrases_by_model[m] = extract_phrases_for_model(
                    x, lbl, m,
                    top_k=top_k,
                    phrase_window=phrase_window,
                    trim_context=trim_context,
                    use_nms=True,
                    nms_thresh=nms_thresh,
                    nms_grab_multiplier=nms_grab_multiplier,
                )

            # compute pairwise metrics
            for a, b in combinations(model_names, 2):
                A = phrases_by_model[a]
                B = phrases_by_model[b]
                rows.append({
                    "sample_idx": ds_idx,
                    "label_idx": lbl,
                    "gt": gt,
                    "model_a": a,
                    "model_b": b,
                    "top1_iou": top1_iou(A, B),
                    "top5_max_iou": topk_max_iou(A, B),
                    "top5_weighted_bestmatch_iou": weighted_best_match_iou(A, B),
                    "top5_unweighted_bestmatch_iou": unweighted_best_match_iou(A, B),
                    "top5_rankaware_bestmatch_iou": rankaware_best_match_iou(A, B, rank_mode="inv", alpha=0.7),
                    "numA": len(A),
                    "numB": len(B),
                })

    df = pd.DataFrame(rows)
    return df

# %%
# Phrase similarity eval restricted to AGREE-POSITIVE (ALL test samples, Q2-style)
# For each (pair, label): scan ALL test samples, keep only agree-positive samples,
# then compute phrase similarity metrics (top-1, top-5 max, top-5 weighted best-match).

# ----------------
# CONFIG (Q2-style)
# ----------------
MODEL_NAMES = ("Centralized", "FedAvg", "FedProx", "SCAFFOLD")
PAIRS = [(a, b) for i, a in enumerate(MODEL_NAMES) for b in MODEL_NAMES[i+1:]]

LABEL_INDICES = list(range(test_dataset[0][1].shape[0]))   # e.g. list(range(50))

TOP_K = 5
PHRASE_WINDOW = 12
NMS_THRESH = 0.30
NMS_GRAB_MULT = 6

MIN_KEPT_REQUIRED = 30     # skip (pair,label) if fewer than this many agree-positive samples

print("Pairs:", PAIRS)
print("Num labels:", len(LABEL_INDICES))
print("Scanning ALL test samples:", len(test_dataset))


# Helper: agree-positive indices for one (pair, label), all_test
@torch.no_grad()
def agree_pos_indices_all_test(model_a: str, model_b: str, label_idx: int) -> list[int]:
    idxs = []
    for ds_idx in range(len(test_dataset)):
        x, _ = test_dataset[ds_idx]
        if bool(agree_positive_mask(model_a, model_b, x.unsqueeze(0), label_idx).item()):
            idxs.append(ds_idx)
    return idxs


# RUN: per pair x per label loop
all_rows = []
coverage_rows = []

for (a, b) in PAIRS:
    print(f"\n==============================")
    print(f"PAIR: {a} vs {b}")
    print("==============================")

    for lbl in tqdm(LABEL_INDICES, desc=f"labels ({a} vs {b})"):
        ap_idxs = agree_pos_indices_all_test(a, b, lbl)

        coverage_rows.append({
            "model_a": a,
            "model_b": b,
            "label_idx": lbl,
            "n_scanned": len(test_dataset),
            "n_agree_pos": len(ap_idxs),
        })

        # Skip low-coverage labels to avoid noisy estimates (same idea as Q2)
        if len(ap_idxs) < MIN_KEPT_REQUIRED:
            continue

        df = run_phrase_similarity_eval(
            sample_indices=ap_idxs,
            label_indices=[lbl],         # IMPORTANT: only this label, so every row is agree-positive by construction
            model_names=(a, b),          # IMPORTANT: only this pair
            top_k=TOP_K,
            phrase_window=PHRASE_WINDOW,
            nms_thresh=NMS_THRESH,
            nms_grab_multiplier=NMS_GRAB_MULT,
        )
        all_rows.append(df)

coverage_df = pd.DataFrame(coverage_rows)
print("\nTop label coverage (by agree-positive):")
display(coverage_df.sort_values(["model_a","model_b","n_agree_pos"], ascending=[True, True, False]).head(20))

df_phrase_sim = pd.concat(all_rows, ignore_index=True) if len(all_rows) else pd.DataFrame()
print("Final agree-positive rows:", len(df_phrase_sim))
display(df_phrase_sim.head())


# Overall summary
METRIC_COLS = [
    "top1_iou",
    "top5_max_iou",
    "top5_weighted_bestmatch_iou",
    "top5_unweighted_bestmatch_iou",
    "top5_rankaware_bestmatch_iou",
]

if len(df_phrase_sim) > 0:
    summary_overall = (
        df_phrase_sim.groupby(["model_a","model_b"])[METRIC_COLS]
        .agg(["mean","std","count"])
        .round(4)
    )
    display(summary_overall)

    # Per-label summary + n  (this creates MultiIndex columns: flatten them next)
    summary_by_label = (
        df_phrase_sim.groupby(["label_idx","model_a","model_b"])[METRIC_COLS]
        .agg(["mean","count"])
    ).reset_index()

    # --- FLATTEN multiindex columns produced by agg(...) into single strings ---
    # example: ('top1_iou', 'mean') -> 'top1_iou_mean'
    flat_cols = []
    for col in summary_by_label.columns:
        if isinstance(col, tuple):
            # join non-empty parts with underscore
            flat_name = "_".join([str(c) for c in col if c != ""]).strip("_")
            flat_cols.append(flat_name)
        else:
            flat_cols.append(str(col))
    summary_by_label.columns = flat_cols

    # --- RENAME flattened metric columns to friendly short names used later ---
    rename_map = {
        "top1_iou_mean": "top1_mean",
        "top1_iou_count": "top1_n",
        "top5_max_iou_mean": "top5max_mean",
        "top5_max_iou_count": "top5max_n",
        "top5_weighted_bestmatch_iou_mean": "wbest_mean",
        "top5_weighted_bestmatch_iou_count": "wbest_n",
        "top5_unweighted_bestmatch_iou_mean": "ubest_mean",
        "top5_unweighted_bestmatch_iou_count": "ubest_n",
        "top5_rankaware_bestmatch_iou_mean": "rankbest_mean",
        "top5_rankaware_bestmatch_iou_count": "rankbest_n",
    }
    summary_by_label = summary_by_label.rename(columns=rename_map)

    # attach true agree-positive coverage (coverage_df already has single-level cols)
    # safe merge now because both sides have single-level columns
    summary_by_label = summary_by_label.merge(
        coverage_df[["model_a","model_b","label_idx","n_agree_pos"]],
        on=["model_a","model_b","label_idx"],
        how="left",
    )

    # Add ICD metadata
    summary_by_label["icd_code"] = summary_by_label["label_idx"].map(lambda i: IDX_TO_CODE.get(int(i), "UNK"))
    summary_by_label["icd_description"] = summary_by_label["icd_code"].map(lambda c: CODE_TO_DESC.get(str(c), "Unknown ICD code"))

    print(f"\nLowest similarity labels (primary metric), filtered to n>={MIN_KEPT_REQUIRED}:")
    interesting = summary_by_label[summary_by_label["n_agree_pos"] >= MIN_KEPT_REQUIRED].sort_values("wbest_mean")
    display(interesting.head(15))

    print(f"\nHighest similarity labels (primary metric), filtered to n>={MIN_KEPT_REQUIRED}:")
    interesting_high = summary_by_label[summary_by_label["n_agree_pos"] >= MIN_KEPT_REQUIRED].sort_values("wbest_mean", ascending=False)
    display(interesting_high.head(15))
else:
    print("No (pair,label) met MIN_KEPT_REQUIRED. Lower MIN_KEPT_REQUIRED or verify agree_positive_mask.")


# %%
# Inspect dissimilar cases and similar cases

if len(df_phrase_sim) > 0:

    assert "sample_idx" in df_phrase_sim.columns, "df_phrase_sim is missing 'sample_idx' column."
    assert "top5_weighted_bestmatch_iou" in df_phrase_sim.columns, "df_phrase_sim missing wbest metric column."

    # Filter once
    filt = summary_by_label[summary_by_label["n_agree_pos"] >= MIN_KEPT_REQUIRED].copy()

    # lowest similarity (label + pair combos)
    lowest_labels = filt.sort_values("wbest_mean").head(3)

    # highest similarity (label + pair combos)
    highest_labels = filt.sort_values("wbest_mean", ascending=False).head(3)

    def pick_extreme_sample(df, label_idx, model_a, model_b, kind="low", metric="top5_weighted_bestmatch_iou"):
        sub = df[
            (df["label_idx"] == label_idx) &
            (df["model_a"] == model_a) &
            (df["model_b"] == model_b)
        ].copy()

        if len(sub) == 0:
            return None

        if kind == "low":
            return sub.sort_values(metric).iloc[0]
        else:
            return sub.sort_values(metric, ascending=False).iloc[0]


    def inspect_combos(combo_df, kind="low", show_all_models=True):
        for _, r in combo_df.iterrows():
            label_idx = int(r["label_idx"])
            model_a   = r["model_a"]
            model_b   = r["model_b"]
            # could also pick via the other metrics
            row = pick_extreme_sample(
                df_phrase_sim, label_idx, model_a, model_b, kind=kind,
                metric="top5_weighted_bestmatch_iou"
            )
            if row is None:
                print(f"Skipping (no rows): label={label_idx}, {model_a} vs {model_b}")
                continue

            sample_idx = int(row["sample_idx"])

            # sample-level metrics
            s_top1     = float(row["top1_iou"])
            s_top5max  = float(row["top5_max_iou"])
            s_wbest    = float(row["top5_weighted_bestmatch_iou"])
            s_ubest    = float(row["top5_unweighted_bestmatch_iou"])
            s_rankbest = float(row["top5_rankaware_bestmatch_iou"])

            print("\n==============================")
            print(f"{kind.upper()} | label_idx={label_idx} | {model_a} vs {model_b}")
            if "icd_code" in r and "icd_description" in r:
                print(f"{r['icd_code']} – {r['icd_description']}")

            # pair-level summary metrics (means) + n
            print(
                "pair means (n): "
                f"top1={float(r['top1_mean']):.4f} (n={int(r['top1_n'])}) | "
                f"top5max={float(r['top5max_mean']):.4f} (n={int(r['top5max_n'])}) | "
                f"wbest={float(r['wbest_mean']):.4f} (n={int(r['wbest_n'])}) | "
                f"ubest={float(r['ubest_mean']):.4f} (n={int(r['ubest_n'])}) | "
                f"rankbest={float(r['rankbest_mean']):.4f} (n={int(r['rankbest_n'])})"
            )

            # sample-level metrics
            print(
                f"chosen sample_idx={sample_idx} | "
                f"sample: top1={s_top1:.4f} | top5max={s_top5max:.4f} | "
                f"wbest={s_wbest:.4f} | ubest={s_ubest:.4f} | rankbest={s_rankbest:.4f}"
            )


            models_to_show = (
                ("Centralized","FedAvg","FedProx","SCAFFOLD")
                if show_all_models
                else (model_a, model_b)
            )

            show_phrase_attention_comparison(
                sample_idx=sample_idx,
                label_idx=label_idx,
                model_names=models_to_show,
                top_k_phrases=5,
                phrase_window=12,
                trim_context=True,
            )

    inspect_combos(lowest_labels, kind="low",  show_all_models=True)
    inspect_combos(highest_labels, kind="high", show_all_models=True)

else:
    print("Skipping inspection because df_phrase_sim is empty.")



