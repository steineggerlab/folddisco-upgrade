"""
04_fit_score.py (최종 학습 원본 복원)

- 구조: MLP (SimpleMLP) + linear skip (wide-and-deep)
- loss: Query group 내 listwise (InfoNCE + max-negative hinge) + Global BCE Calibration
- feature contract 및 정규화 통계(mu, sd) 저장
"""

import os
import gc
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from folddisco_features import (FEATURE_NAMES, ALL_FEATURES, normalize_columns,
                                build_features, group_size_guard, per_query_z)

RESULT_DIR = "result_benchmark_v2"
ANALYSES = os.path.join(RESULT_DIR, "folddisco_results_analyses")
MCSA_PATH = os.path.join(ANALYSES, "filtered_combined_MCSA.txt")
MODEL_SAVE_PATH = LOG_FILE = None   # args 파싱 후 설정

p = argparse.ArgumentParser()
p.add_argument("--data", default=MCSA_PATH)
p.add_argument("--query_key", default="source", help="M-CSA 는 source 가 query (scope40 과 반대)")
p.add_argument("--dedupe", action="store_true", help="(source, pdb_basename) 중복 제거")
p.add_argument("--arch", choices=["mlp", "linear"], default="mlp")
p.add_argument("--skip", choices=["none", "linear"], default="linear",
               help="linear: score = mlp(x) + linear(x) (wide-and-deep).")
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--patience", type=int, default=8)
p.add_argument("--groups_per_step", type=int, default=8)
p.add_argument("--negs_per_group", type=int, default=2000)
p.add_argument("--lr", type=float, default=3e-3)
p.add_argument("--lam_hinge", type=float, default=1.0)
p.add_argument("--beta_bce", type=float, default=1.0,
               help="query 간 캘리브레이션 항.")
p.add_argument("--bce_batch", type=int, default=8192)
p.add_argument("--features", default=",".join(FEATURE_NAMES),
               help=f"쉼표 구분. 가능: {','.join(ALL_FEATURES)}")
p.add_argument("--afdb", default="", help="03_prep_afdb.py 가 만든 .npz. BCE 항에만 쓰인다")
p.add_argument("--afdb_frac", type=float, default=0.5,
               help="BCE 배치에서 AFDB 가 차지할 비율")
p.add_argument("--val_frac", type=float, default=0.15)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--score_dir", default="folddisco_results_scores_final2")
p.add_argument("--model_out", default="", help="비우면 score_dir/best_mlp_model.pt")
args = p.parse_args()

SCORE_DIR = os.path.join(RESULT_DIR, args.score_dir)
os.makedirs(SCORE_DIR, exist_ok=True)
MODEL_SAVE_PATH = args.model_out or os.path.join(SCORE_DIR, "best_mlp_model.pt")
LOG_FILE = os.path.join(SCORE_DIR, "fit_log.txt")


def log(m):
    print(m, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(m + "\n")


def map_vectorized(qc, y, s):
    """query 별 AP 평균. sklearn average_precision_score 와 동일한 동점 처리."""
    s = np.where(np.isfinite(s), s, -np.inf)
    order = np.lexsort((-s, qc))
    q_s, y_s, s_s = qc[order], y[order].astype(np.float64), s[order]
    starts = np.flatnonzero(np.r_[True, q_s[1:] != q_s[:-1]])
    counts = np.diff(np.r_[starts, len(q_s)])
    gidx = np.repeat(np.arange(len(starts)), counts)
    cum_tp = np.cumsum(y_s)
    cum_tp -= np.repeat(np.r_[0.0, cum_tp[starts[1:] - 1]], counts)
    cum_n = np.arange(len(q_s)) - np.repeat(starts, counts) + 1.0
    is_last = np.r_[np.r_[True, q_s[1:] != q_s[:-1]][1:], True]
    be = np.flatnonzero(is_last | np.r_[s_s[1:] != s_s[:-1], True])
    n_pos = np.add.reduceat(y_s, starts)
    valid = (n_pos > 0) & (n_pos < counts)
    prec, rec = cum_tp / cum_n, cum_tp / np.maximum(np.repeat(n_pos, counts), 1e-12)
    g, pb, rb = gidx[be], prec[be], rec[be]
    prev = np.r_[0.0, rb[:-1]]
    prev[np.r_[True, g[1:] != g[:-1]]] = 0.0
    ap = np.bincount(g, weights=(rb - prev) * pb, minlength=len(starts))
    return float(ap[valid].mean()) if valid.any() else float("nan")


# ============================ 로드 ============================
log("=" * 80)
log(f"[load] {args.data}")
t0 = time.time()
df = pd.read_csv(args.data, sep="\t", engine="c",
                 dtype={"query_num": str, "source": str, "target_id": str, "id": str})
df = normalize_columns(df)
for c in ("query_num", "source", "target_id"):
    if c in df.columns:
        df[c] = df[c].astype(str).str.replace("\r", "", regex=False)
log(f"[load] rows={len(df)}  ({time.time()-t0:.1f}s)")

if args.dedupe:
    n0 = len(df)
    df["_pdb"] = df["target_id"].str.rsplit("/", n=1).str[-1]
    df = (df.sort_values("label_raw", ascending=False)
            .drop_duplicates([args.query_key, "_pdb"], keep="first")
            .drop(columns=["_pdb"]).reset_index(drop=True))
    log(f"[dedupe] {n0} -> {len(df)}  ({100*(1-len(df)/n0):.1f}% 제거)")

df = df[df["idf"].notna() & df["tm_score"].notna() & (df["L"] > 1) & (df["L"] <= 32)]
df = df.reset_index(drop=True)
if "rmsd" not in df.columns:
    df["rmsd"] = 999.0
df["rmsd"] = df["rmsd"].fillna(999.0)

y = np.where(df["label_raw"].to_numpy() == 0, 0, 1).astype(np.int8)
FEATS = [f.strip() for f in args.features.split(",") if f.strip()]
X, qc, nq = build_features(df, args.query_key, FEATS)
log(group_size_guard(qc, nq, "M-CSA", reference=9228))
log(f"[data] rows={len(df)}  groups={nq}  positives={int(y.sum())} ({100*y.mean():.3f}%)")
log(f"[feat] {len(FEATS)}: {FEATS}")
del df
gc.collect()

# ---------- AFDB 캐시 (BCE 항 전용) ----------
AF_X = AF_y = None
if args.afdb:
    _z = np.load(args.afdb, allow_pickle=True)
    _names = [str(v) for v in _z["feature_names"]]
    _missing = [f for f in FEATS if f not in _names]
    if _missing:
        raise SystemExit(
            f"[FATAL] AFDB 캐시에 없는 feature: {_missing}\n"
            f"        캐시: {_names}\n        현재: {FEATS}\n"
            f"        해결: python 03_prep_afdb.py --features {','.join(FEATS)}")
    _sel = [_names.index(f) for f in FEATS]
    AF_X, AF_y = _z["X"][:, _sel].astype(np.float32), _z["y"].astype(np.int8)
    if _names != FEATS:
        log(f"[afdb] 캐시 {len(_names)}개 중 {len(FEATS)}개 선택: {FEATS}")
    _g = _z["gid"]
    log(f"[afdb] {len(AF_X)} rows, {len(np.unique(_g))} groups, "
        f"positives={int(AF_y.sum())} (negative-only 예상)")
    log("[afdb] listwise 항에는 기여 불가(positive 없음). BCE 항에만 사용한다.")

# ============================ 모델 ============================
class LinearScorer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 1)

    def forward(self, x):
        return self.lin(x).squeeze(-1)


class MLPScorer(nn.Module):
    def __init__(self, d, h=256, skip="none"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(),
            nn.Linear(h, h // 2), nn.LayerNorm(h // 2), nn.ReLU(),
            nn.Linear(h // 2, 1))
        self.skip = nn.Linear(d, 1) if skip == "linear" else None
        if self.skip is not None:
            nn.init.zeros_(self.skip.weight)
            nn.init.zeros_(self.skip.bias)

    def forward(self, x):
        s = self.net(x).squeeze(-1)
        if self.skip is not None:
            s = s + self.skip(x).squeeze(-1)
        return s


def listwise_loss(s, yb, lam):
    pos, neg = s[yb == 1], s[yb == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    lse_neg = torch.logsumexp(neg, 0)
    l_nce = (torch.logaddexp(pos, lse_neg.expand_as(pos)) - pos).mean()
    l_ffp = F.softplus(lse_neg - pos).mean()
    return l_nce + lam * l_ffp


# ============================ 학습 ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)

order = np.argsort(qc, kind="stable")
gstart = np.searchsorted(qc[order], np.arange(nq))
gend = np.searchsorted(qc[order], np.arange(nq), side="right")
gidx_rows = [order[gstart[g]:gend[g]] for g in range(nq)]
gpos = [r[y[r] == 1] for r in gidx_rows]
gneg = [r[y[r] == 0] for r in gidx_rows]
usable = np.flatnonzero([len(gpos[g]) > 0 and len(gneg[g]) > 0 for g in range(nq)])
log(f"[data] usable groups: {len(usable)}/{nq}")

rng.shuffle(usable)
ncut = max(int(args.val_frac * len(usable)), 1)
va_g, tr_g = usable[:ncut], usable[ncut:]
log(f"[split] train={len(tr_g)} groups, val={len(va_g)} groups")

# 표준화: train group 행으로만
rows_tr = np.concatenate([gidx_rows[g] for g in tr_g])
mu = X[rows_tr].mean(0, keepdims=True)
sd = X[rows_tr].std(0, keepdims=True) + 1e-6
X = ((X - mu) / sd).astype(np.float32)
if AF_X is not None:
    AF_X_SCALED = ((AF_X - mu) / sd).astype(np.float32)
    del AF_X
    AF_X = AF_X_SCALED
del rows_tr
gc.collect()

rows_va = np.concatenate([gidx_rows[g] for g in va_g])
Xva, yva = X[rows_va], y[rows_va]
qva = pd.factorize(qc[rows_va], sort=False)[0].astype(np.int64)

model = (MLPScorer(X.shape[1], skip=args.skip) if args.arch == "mlp"
         else LinearScorer(X.shape[1])).to(device)
opt = torch.optim.AdamW(model.parameters(),
                        lr=(args.lr if args.arch == "mlp" else 3e-2), weight_decay=1e-4)
log(f"[model] {args.arch}  params={sum(p.numel() for p in model.parameters())}  beta_bce={args.beta_bce}  skip={args.skip}")


def predict(m, Xs, bs=1_000_000):
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xs), bs):
            out.append(m(torch.from_numpy(Xs[i:i + bs]).to(device))
                       .float().cpu().numpy().astype(np.float64))
    return np.concatenate(out)


best, best_state, bad = -1.0, None, 0
for ep in range(1, args.epochs + 1):
    model.train()
    perm = rng.permutation(tr_g)
    tot, nb = 0.0, 0
    for i in range(0, len(perm), args.groups_per_step):
        losses = []
        for g in perm[i:i + args.groups_per_step]:
            pr, ng = gpos[g], gneg[g]
            if len(ng) > args.negs_per_group:
                ng = rng.choice(ng, args.negs_per_group, replace=False)
            rows = np.concatenate([pr, ng])
            xb = torch.from_numpy(X[rows]).to(device)
            yb = torch.from_numpy(y[rows].astype(np.int64)).to(device)
            l = listwise_loss(model(xb), yb, args.lam_hinge)
            if l is not None:
                losses.append(l)
        if not losses:
            continue
        loss = torch.stack(losses).mean()

        if args.beta_bce > 0:
            n_af = int(args.bce_batch * args.afdb_frac) if AF_X is not None else 0
            n_mc = args.bce_batch - n_af
            bi = rng.choice(len(X), min(n_mc, len(X)), replace=False)
            xs, ys = [X[bi]], [y[bi].astype(np.float32)]
            if n_af > 0:
                ai = rng.choice(len(AF_X), n_af, replace=(n_af > len(AF_X)))
                xs.append(AF_X_SCALED[ai]); ys.append(AF_y[ai].astype(np.float32))
            xb2 = torch.from_numpy(np.concatenate(xs)).to(device)
            yb2 = torch.from_numpy(np.concatenate(ys)).to(device)
            pw = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)],
                              dtype=torch.float32, device=device)
            loss = loss + args.beta_bce * F.binary_cross_entropy_with_logits(
                model(xb2), yb2, pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss); nb += 1

    _sv = predict(model, Xva)
    m = map_vectorized(qva, yva, _sv)
    _gap_raw = average_precision_score(yva, _sv)
    _gap_z = average_precision_score(yva, per_query_z(qva, _sv, qva.max() + 1))
    if m > best:
        best, bad = m, 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
    log(f"Epoch {ep:02d}/{args.epochs} - Loss: {tot/max(nb,1):.4f}  val_mAP: {m:.4f} "
        f"(best {best:.4f})  val_globalAP: raw={_gap_raw:.4f} z={_gap_z:.4f}")
    if bad >= args.patience:
        log(f"[early stop] @ep{ep}")
        break

model.load_state_dict(best_state)
if args.arch == "linear":
    w = model.lin.weight.detach().cpu().numpy().ravel()
    log("[weights] " + "  ".join(f"{n}={v:+.3f}" for n, v in zip(FEATS, w)))
if getattr(model, "skip", None) is not None:
    w = model.skip.weight.detach().cpu().numpy().ravel()
    log("[skip weights] " + "  ".join(f"{n}={v:+.3f}" for n, v in zip(FEATS, w)))

# ============================ 저장 ============================
torch.save({
    "state_dict": best_state,
    "feature_names": FEATS,
    "arch": args.arch,
    "skip": args.skip,
    "in_dim": X.shape[1],
    "hidden": 256,
    "mu": mu, "sd": sd,
    "query_key_train": args.query_key,
    "val_mAP": best,
    "beta_bce": args.beta_bce,
}, MODEL_SAVE_PATH)
log(f"\n[save] {MODEL_SAVE_PATH}")
log(f"[save] feature contract: {FEATS}")
log(f"[save] val globalAP: raw={_gap_raw:.4f}  z={_gap_z:.4f}")
log(f"[save] best val_mAP = {best:.4f}")
log("=" * 80)