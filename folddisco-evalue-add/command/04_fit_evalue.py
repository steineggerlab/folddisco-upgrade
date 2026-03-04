"""
Pseudo-code (but close to runnable) for training a simple MLP on four inputs:
  - length (computed as number of matched residues from `matching_residues`)
  - idf_score_per_match
  - rmsd
  - tm_score

and converting the model output into per-match E-values using an empirical null tail.

Columns (tab-separated):
  query_num, source, id, node_count, idf_score_per_match, rmsd, tm_score,
  matching_residues, self_similar_label

Label convention in your example:
  2 = self
  1 = similar
  0 = non-similar
"""

import os
import math
import numpy as np
from collections import defaultdict

# ---------- 1) Parsing & feature construction ----------

INPUT_PATH = "result/folddisco_results_analyses/total_combined.txt"
EVALUE_DIR = "result/folddisco_results_evalues"
OUTPUT_PATH = f"{EVALUE_DIR}/total_evalues_test.txt"
LOG_FILE = f"{EVALUE_DIR}/fit_log.txt"

os.makedirs(EVALUE_DIR, exist_ok=True)

def log_and_print(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def count_matching_residues(matching_residues_str: str) -> int:
    """
    Example field:
      "C6,C10,C11,..." or "_,_,A143,_,A253,..."
    We interpret length as the count of non-'_' tokens.
    """
    tokens = matching_residues_str.strip().split(",")
    # Some tokens may be "" if there are trailing commas; ignore them.
    valid = [t for t in tokens if t and t != "_"]
    return len(valid)

def safe_float(x: str) -> float:
    try:
        return float(x)
    except:
        return float("nan")

def load_rows(path: str):
    rows = []
    with open(path, "r") as f:
        header = f.readline().strip().split()
        # Expect tab/space separated; your example looks tabbed but split() covers both.

        # Map column name -> index (robust to order)
        col = {name: i for i, name in enumerate(header)}

        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            # If the file is space-separated instead of tab-separated:
            if len(parts) == 1:
                parts = line.strip().split()

            q = parts[col["query_num"]]
            tgt = parts[col["id"]]
            idf = safe_float(parts[col["idf_score_per_match"]])
            rmsd = safe_float(parts[col["rmsd"]])
            tm_score = safe_float(parts[col["tm_score"]])
            matching = parts[col["matching_residues"]]
            lab = int(parts[col["self_similar_label"]])

            L = count_matching_residues(matching)

            # Drop pathological rows (optional)
            if any(math.isnan(v) for v in [idf, rmsd, tm_score]) or L <= 0 or L > 32:
                continue

            rows.append({
                "query_num": q,
                "target_id": tgt,
                "L": L,
                "idf": idf / max(L, 1),  # idf per match
                "rmsd": rmsd,
                "tm_score": tm_score,
                "label_raw": lab,
            })
    return rows

rows = load_rows(INPUT_PATH)
log_and_print(f"Loaded {len(rows)} rows from {INPUT_PATH}")

# ---------- 2) Label mapping ----------
# Goal: binary classifier P(homologous) where homologous = self or similar
# raw labels: 0=non-similar, 1=similar, 2=self (assumed)
def to_binary_label(label_raw: int) -> int:
    return 0 if label_raw == 0 else 1

for r in rows:
    r["y"] = to_binary_label(r["label_raw"])

# ---------- 3) Train/val/test split (avoid leakage) ----------
# You *should* split by CATH superfamily, but your file doesn't include it.
# So below is a placeholder "group split by query_num" to avoid leaking the same query across splits.
# Replace this with superfamily-disjoint splitting when you can join CATH IDs.
def split_by_query(rows, seed=0, frac_train=0.8, frac_val=0.1):
    rng = np.random.default_rng(seed)
    queries = sorted({r["query_num"] for r in rows})
    rng.shuffle(queries)

    n = len(queries)
    n_train = int(frac_train * n)
    n_val = int(frac_val * n)

    train_q = set(queries[:n_train])
    val_q   = set(queries[n_train:n_train+n_val])
    test_q  = set(queries[n_train+n_val:])

    train = [r for r in rows if r["query_num"] in train_q]
    val   = [r for r in rows if r["query_num"] in val_q]
    test  = [r for r in rows if r["query_num"] in test_q]
    return train, val, test

train_rows, val_rows, test_rows = split_by_query(rows, seed=42)
log_and_print(f"Train: {len(train_rows)} rows, Val: {len(val_rows)}, Test: {len(test_rows)}")

# ---------- 4) Feature transform / standardization ----------
# We only use four inputs: length, IDF, RMSD, TM.
# We will apply robust transforms that often help:
#   - length: log1p(L)
#   - idf: log1p(idf)
#   - rmsd: log1p(rmsd)   (or clip then log1p)
#   - tm: keep as-is (already bounded-ish 0..1), or use logit-ish transform if desired

def featurize(r):
    x_len = math.log1p(r["L"])
    x_idf = math.log1p(max(r["idf"], 0.0))
    x_rmsd = math.log1p(max(r["rmsd"], 0.0))
    x_tm_score = r["tm_score"]
    return np.array([x_len, x_idf, x_rmsd, x_tm_score], dtype=np.float32)

X_train = np.stack([featurize(r) for r in train_rows])
y_train = np.array([r["y"] for r in train_rows], dtype=np.float32)

X_val = np.stack([featurize(r) for r in val_rows])
y_val = np.array([r["y"] for r in val_rows], dtype=np.float32)

X_test = np.stack([featurize(r) for r in test_rows])
y_test = np.array([r["y"] for r in test_rows], dtype=np.float32)

# Standardize features based on train only
mu = X_train.mean(axis=0)
sd = X_train.std(axis=0) + 1e-6

def standardize(X):
    return (X - mu) / sd

X_train = standardize(X_train)
X_val = standardize(X_val)
X_test = standardize(X_test)

log_and_print(f"Feature means (train): {mu}")
log_and_print(f"Feature stds (train): {sd}")

# ---------- 5) Simple MLP model (PyTorch-like pseudo-code) ----------
# You can implement in PyTorch, JAX, or TF. Below uses PyTorch-style syntax.

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

class SimpleMLP(nn.Module):
    """
    4 -> hidden -> hidden -> 1 (logit)
    Keep it small; tabular problems often work well with modest depth.
    """
    def __init__(self, in_dim=4, hidden=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, 1)  # output logit
        )

    def forward(self, x):
        # x: (B,4)
        return self.net(x).squeeze(-1)  # (B,)

model = SimpleMLP(in_dim=4, hidden=32, dropout=0.1)

# Class imbalance handling: pos_weight for BCEWithLogitsLoss
pos = float((y_train == 1).sum())
neg = float((y_train == 0).sum())
pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

# Mini-batch loader
def batches(X, y, batch_size=2048, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    for i in range(0, len(X), batch_size):
        j = idx[i:i+batch_size]
        yield torch.tensor(X[j], dtype=torch.float32), torch.tensor(y[j], dtype=torch.float32)

# Train loop
def train_model(model, X_train, y_train, X_val, y_val, epochs=10):
    best_val = float("inf")
    best_state = None

    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        n = 0

        for xb, yb in batches(X_train, y_train, batch_size=2048, seed=ep):
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(xb)
            n += len(xb)

        # validation
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(X_val, dtype=torch.float32)
            yv = torch.tensor(y_val, dtype=torch.float32)
            lv = criterion(model(xv), yv).item()

        train_loss = running / max(n, 1)
        val_loss = lv

        log_and_print(f"epoch {ep:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

train_model(model, X_train, y_train, X_val, y_val, epochs=20)
log_and_print("Training complete.")

# ---------- 6) Test evaluation and scoring ----------

# Collect logits for the training set to use for scaling if necessary
model.eval()
with torch.no_grad():
    logits_train = model(torch.tensor(X_train, dtype=torch.float32)).cpu().numpy()

# Predict on test data and compute the probability score (0 to 1)

header = "query_num\ttarget_id\tL\tidf\trmsd\ttm_score\tlabel_raw\tscore_logit\tprob_homologous\n"
with open(OUTPUT_PATH, "w") as f:
    f.write(header)

for row in test_rows:
    X_test_row = featurize(row)  # featureize the test row
    X_test_row = standardize(X_test_row.reshape(1, -1))  # standardize the features
    
    # Get the logit (raw model output) and probability (using sigmoid)
    logit_tensor = model(torch.tensor(X_test_row, dtype=torch.float32))  # get logit as a scalar
    prob_homologous = torch.sigmoid(logit_tensor).item()  # probability using sigmoid
    logit = logit_tensor.item()

    # Save results for each test row
    with open(OUTPUT_PATH, "a") as f:
        f.write(f"{row['query_num']}\t{row['target_id']}\t{row['L']}\t{row['idf']}\t{row['rmsd']}\t{row['tm_score']}\t{row['label_raw']}\t{logit:.6f}\t{prob_homologous:.6f}\n")

log_and_print(f"Saved scored results to {OUTPUT_PATH}")

log_and_print("Computation complete.")

# ---------- 10) Notes / upgrades (still simple) ----------
# - If you can join CATH superfamily IDs, you should do superfamily-disjoint splitting.
# - You can compute null tails from a dedicated "non-similar-only" corpus that matches your real search.
# - If you want *extremely* small p-values, replace empirical tail with a smoothed tail fit (e.g., isotonic on rank or GPD on extremes).
# - If you want separate self vs similar, add a second head and train multi-task:
#     head1: homologous vs non
#     head2: self vs similar among positives