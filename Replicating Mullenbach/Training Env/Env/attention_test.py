# %% [markdown]
# # Attention Notebook to Test If All the Models Focus on the Same Words

# %% [markdown]
# ## Set Up from Previous

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


# %%
config = {
    "batch_size": 32,
    "lr": 0.002,
    "n_filters": 21,
    "window_size": 6,
    # the rest of the keys aren't used in the attention notebook
}

model_param_path = os.path.join("..", "Model", "processed_full.w2v")

# %% [markdown]
# ### Recover Tokens from Input Sequences

# %%
from gensim.models import Word2Vec

# Load your W2V model to get the mapping index → token
w2v = Word2Vec.load(model_param_path)
idx_to_token = w2v.wv.index_to_key   # list where index maps to actual word

PAD_INDEX = len(idx_to_token)  # from ConvAttnPool padding_idx


# %%
# Load Trained Models and Test Data

# Paths to the saved models (update if your filenames differ)
MODEL_PATHS = {
    "Centralized": "../History/models/central_best_attention.pt",
    "FedAvg":      "../History/models/fedavg_c2e3_best_attention.pt",
    "FedProx":     "../History/models/fedprox_c2e3_best_attention.pt",
    "SCAFFOLD":    "../History/models/scaffold_c2e3_best_attention.pt",
}

def load_model(path):
    model = GenerateModel(
        table_path=model_param_path,
        num_of_filters=config["n_filters"],
        kernel_size=config["window_size"],
    ).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

# Load all four models
models = {name: load_model(p) for name, p in MODEL_PATHS.items()}
print("Loaded models:", list(models.keys()))

# Load test set
X_test = torch.load(os.path.join("..", "Data", "X_test.pt"))
Y_test = torch.load(os.path.join("..", "Data", "Y_test.pt"))
test_dataset = TensorDataset(X_test, Y_test)
print("Test samples:", len(test_dataset))


# %% [markdown]
# ## Choose Labels and Gather Shared Positive Samples

# %%
def collect_positive_samples_for_labels(dataset, label_indices, n_per_label=5):
    label_indices = list(label_indices)
    idx_by_label = {lbl: [] for lbl in label_indices}

    for i, (_, y) in enumerate(dataset):
        y = y.long()
        for lbl in label_indices:
            if y[lbl] == 1 and len(idx_by_label[lbl]) < n_per_label:
                idx_by_label[lbl].append(i)
        if all(len(v) >= n_per_label for v in idx_by_label.values()):
            break

    return idx_by_label

def build_batch_from_indices(dataset, indices):
    X_list, Y_list = [], []
    for idx in indices:
        x, y = dataset[idx]
        X_list.append(x)
        Y_list.append(y)
    return torch.stack(X_list), torch.stack(Y_list)

# pick some label indices you care about
label_indices = [3, 7]     # example; change to interesting labels
N_PER_LABEL   = 200        # <-- average over many positives (raise/lower as needed)

sample_indices_by_label = collect_positive_samples_for_labels(
    test_dataset, label_indices, n_per_label=N_PER_LABEL
)
sample_indices_by_label


# %% [markdown]
# ## Extract Attention for All Models and Labels

# %%
@torch.no_grad()
def get_attention_for_label(model, X_batch, label_idx):
    X_batch = X_batch.to(device)
    logits, alpha = model(X_batch)  # alpha: (B, label_space, seq_len)
    # attention for this label only
    alpha_label = alpha[:, label_idx, :].cpu()   # (B, seq_len)
    logits_label = logits[:, label_idx].cpu()
    return logits_label, alpha_label

attention_results = {}  # {model_name: {label_idx: {indices, logits, alpha}}}

for model_name, model in models.items():
    attention_results[model_name] = {}
    for lbl in label_indices:
        idxs = sample_indices_by_label[lbl]
        X_batch, Y_batch = build_batch_from_indices(test_dataset, idxs)
        logits_lbl, alpha_lbl = get_attention_for_label(model, X_batch, lbl)
        attention_results[model_name][lbl] = {
            "indices": idxs,
            "logits": logits_lbl,
            "alpha": alpha_lbl,   # shape (B, seq_len)
        }

# quick sanity check
for m in attention_results:
    for lbl in attention_results[m]:
        print(m, "label", lbl, "alpha shape:",
              attention_results[m][lbl]["alpha"].shape)


# %% [markdown]
# ## Quantative

# %% [markdown]
# ###  Simple attention visualization (position heatmaps)

# %%
def plot_attention_heatmap(alpha_vec, title=""):
    plt.figure(figsize=(10, 1.5))
    plt.imshow(alpha_vec.unsqueeze(0).numpy(), aspect="auto")
    plt.colorbar(label="Attention weight")
    plt.yticks([])
    plt.xlabel("Token position")
    plt.title(title)
    plt.show()

# Example: first sample for first label, across all models
target_label = label_indices[0]
sample_in_batch = 0

for model_name in models.keys():
    alpha_batch = attention_results[model_name][target_label]["alpha"]
    alpha_vec = alpha_batch[sample_in_batch]  # (seq_len,)
    plot_attention_heatmap(
        alpha_vec,
        title=f"{model_name} – label {target_label} – sample {sample_in_batch}",
    )


# %% [markdown]
# ### Similarity Metrics Between Models

# %%
import torch.nn.functional as F

def cosine_sim(a, b):
    # a, b: (seq_len,)
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

model_names = list(models.keys())

for lbl in label_indices:
    print(f"\n=== Label {lbl} ===")
    idx0 = sample_indices_by_label[lbl][0]
    X_batch, _ = build_batch_from_indices(test_dataset, [idx0])

    # collect one attention vector per model
    alpha_single = {}
    for name, model in models.items():
        _, alpha_lbl = get_attention_for_label(model, X_batch, lbl)
        alpha_single[name] = alpha_lbl[0]  # (seq_len,)

    # build similarity matrix
    sims = np.zeros((len(model_names), len(model_names)))
    for i, name_i in enumerate(model_names):
        for j, name_j in enumerate(model_names):
            sims[i, j] = cosine_sim(alpha_single[name_i], alpha_single[name_j])

    sim_df = pd.DataFrame(sims, index=model_names, columns=model_names)
    print(f"Cosine similarity of attention vectors for label {lbl}, sample idx {idx0}:")
    display(sim_df)


# %%
# --- Jaccard similarity helpers (PAD + context-window trimming) ---

def get_nonpad_len(x_1d: torch.Tensor, pad_index: int) -> int:
    """
    Returns number of non-PAD tokens assuming PAD fills from the end.
    """
    x = x_1d.detach().cpu()
    pad_pos = (x == pad_index).nonzero(as_tuple=False)
    return int(pad_pos[0].item()) if len(pad_pos) > 0 else int(x.numel())

def get_valid_positions(x_1d: torch.Tensor, pad_index: int, window_size: int | None) -> np.ndarray:
    """
    Valid attention positions:
      - exclude PAD region
      - optionally trim 'half window' tokens at both ends to reduce conv padding artifacts
    """
    L = get_nonpad_len(x_1d, pad_index)
    if L <= 0:
        return np.array([], dtype=int)

    left, right = 0, L  # right is exclusive
    if window_size is not None and window_size > 1:
        half = window_size // 2
        left = min(left + half, L)
        right = max(right - half, left)

    return np.arange(left, right, dtype=int)

def topk_positions(att_vec_1d: torch.Tensor, valid_pos: np.ndarray, k: int) -> set[int]:
    """
    Return set of token positions corresponding to top-k attention weights within valid_pos.
    """
    if valid_pos.size == 0:
        return set()

    att = att_vec_1d.detach().cpu().numpy()
    att_valid = att[valid_pos]
    if att_valid.size == 0:
        return set()

    k_eff = min(k, att_valid.size)
    idx_part = np.argpartition(att_valid, -k_eff)[-k_eff:]
    idx_sorted = idx_part[np.argsort(att_valid[idx_part])[::-1]]

    return set(valid_pos[idx_sorted].tolist())

def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# %%
# --- Average Jaccard similarity across many positive samples (per label) ---

@torch.no_grad()
def attention_vec_for_label_single(model, x_1d, label_idx):
    """
    x_1d: (seq_len,)
    returns: attention vec (seq_len,)
    """
    _, alpha_lbl = get_attention_for_label(model, x_1d.unsqueeze(0), label_idx)
    return alpha_lbl[0]  # (seq_len,)

def avg_jaccard_matrix_for_label(
    dataset,
    sample_indices: list[int],
    label_idx: int,
    top_k: int = 15,
    trim_context_window: bool = True,
):
    model_names = list(models.keys())
    mat_vals = { (a,b): [] for a in model_names for b in model_names }

    window_size = config["window_size"] if trim_context_window else None

    for ds_idx in sample_indices:
        x, _ = dataset[ds_idx]
        valid_pos = get_valid_positions(x, PAD_INDEX, window_size)

        topk_by_model = {}
        for mname, m in models.items():
            att = attention_vec_for_label_single(m, x.to(device), label_idx)
            topk_by_model[mname] = topk_positions(att, valid_pos, k=top_k)

        for a in model_names:
            for b in model_names:
                mat_vals[(a,b)].append(jaccard(topk_by_model[a], topk_by_model[b]))

    # build dataframe
    M = np.zeros((len(model_names), len(model_names)), dtype=float)
    for i, a in enumerate(model_names):
        for j, b in enumerate(model_names):
            vals = mat_vals[(a,b)]
            M[i, j] = float(np.mean(vals)) if len(vals) else np.nan

    return pd.DataFrame(M, index=model_names, columns=model_names)

def mean_off_diagonal(df: pd.DataFrame) -> float:
    arr = df.values
    mask = ~np.eye(arr.shape[0], dtype=bool)
    return float(np.nanmean(arr[mask]))

TOP_K = 15
TRIM_CONTEXT = True  # set False to keep edge tokens

jaccard_results = {}
summary_rows = []

for lbl in label_indices:
    idxs = sample_indices_by_label[lbl]
    df_j = avg_jaccard_matrix_for_label(
        dataset=test_dataset,
        sample_indices=idxs,
        label_idx=lbl,
        top_k=TOP_K,
        trim_context_window=TRIM_CONTEXT,
    )
    jaccard_results[lbl] = df_j

    print(f"\nAvg Jaccard (top-{TOP_K}) | label={lbl} | n={len(idxs)} | trim_context={TRIM_CONTEXT}")
    display(df_j)

    summary_rows.append({
        "label": lbl,
        "n_samples": len(idxs),
        "top_k": TOP_K,
        "trim_context": TRIM_CONTEXT,
        "mean_pairwise_jaccard": mean_off_diagonal(df_j),
    })

summary_df = pd.DataFrame(summary_rows).sort_values("mean_pairwise_jaccard", ascending=False)
print("\nSummary (mean off-diagonal Jaccard per label):")
display(summary_df)


# %% [markdown]
# ## Qualitative

# %% [markdown]
# ### Decode a Sequence of Token IDs into Words

# %%
def decode_sequence(token_ids):
    tokens = []
    for idx in token_ids:
        idx = int(idx)
        if idx == PAD_INDEX:
            tokens.append("<PAD>")
        else:
            tokens.append(idx_to_token[idx])
    return tokens


# %% [markdown]
# ### Visualize Attention By Token

# %%
def get_top_tokens(tokens, attention, k=15, skip_pad=True):
    """
    Return the top-k tokens by attention weight.
    tokens   : list[str]
    attention: 1D tensor of attention weights
    k        : how many tokens to return
    skip_pad : skip '<PAD>' tokens
    """
    att = attention.cpu().numpy()
    idxs = np.argsort(att)[::-1]  # highest first

    top = []
    for idx in idxs:
        tok = tokens[idx]
        if skip_pad and tok == "<PAD>":
            continue
        top.append((idx, tok, float(att[idx])))
        if len(top) >= k:
            break
    return top


# %%
from IPython.display import HTML

def attention_to_html(tokens, attention, max_color=240):
    att = attention / attention.max()
    html = ""
    for tok, w in zip(tokens, att):
        intensity = int(max_color * float(w))
        html += f"<span style='background-color: rgb(255,255,{255-intensity}); padding:2px; margin:1px;'>{tok}</span>"
    return HTML(html)

def show_attention(model_name, label_idx, sample_dataset_idx=0, top_k=15):
    # get the sample
    x, y = test_dataset[sample_dataset_idx]
    tokens = decode_sequence(x)

    # get attention weights for this model + label
    logits, alpha = get_attention_for_label(models[model_name], x.unsqueeze(0), label_idx)
    attention_vec = alpha[0]  # shape (seq_len,)

    print(f"\nModel: {model_name} | Label: {label_idx} | Sample index: {sample_dataset_idx}")
    display(attention_to_html(tokens, attention_vec))

    # print top-k tokens by attention
    top = get_top_tokens(tokens, attention_vec, k=top_k)
    print(f"Top {top_k} tokens by attention:")
    for pos, tok, w in top:
        print(f"  pos={pos:4d}  att={w:.4f}  token='{tok}'")


# %% [markdown]
# ### Compare Qualitative Attention Across Models

# %%
target_label = 7          # ← change to any interesting label
sample_idx    = 123       # ← a test sample index

for model_name in models.keys():
    show_attention(model_name, target_label, sample_idx)



