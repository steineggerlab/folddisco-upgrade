"""
07_listwise_ab.py

같은 feature / 같은 listwise loss / 같은 5-fold split 에서
  A. 로지스틱 회귀 (선형, 파라미터 ~8개)
  B. MLP (기존 구조)
를 공정하게 붙인다. 기준선(IDF, IDF_then_RMSD)도 동일 fold 에서 계산한다.

설계 근거 (진단 결과 반영):
  - group key = 'source' 단독 (unique source == unique (query_num,source) == 751)
  - query motif 길이는 L 이 아니라 query_num 의 length_N (L 은 매칭 크기)
  - 파일은 CRLF, group 별 CONTIGUOUS
  - 행의 97.6% 가 idf 동점 블록 -> 동점 해소가 랭킹 문제 그 자체
  - mixed-label 동점 블록만 쓰면 큰 블록에 편향되므로, 학습은 전체 group listwise 로 하고
    idf+eps*g 는 추론 시에만 적용한다
  - 751 group 뿐이라 단일 split 은 분산이 큼 -> 5-fold CV 필수

usage:
  python 07_listwise_ab.py --dedupe
  python 07_listwise_ab.py --dedupe --folds 5 --epochs 30
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

RESULT_DIR = "result_benchmark_v2"
ANALYSES = os.path.join(RESULT_DIR, "folddisco_results_analyses")
OUT_DIR = os.path.join(RESULT_DIR, "folddisco_results_scores_test")
MCSA_PATH = os.path.join(ANALYSES, "filtered_combined_MCSA.txt")
LOG_FILE = os.path.join(OUT_DIR, "listwise_ab_log.txt")
os.makedirs(OUT_DIR, exist_ok=True)

p = argparse.ArgumentParser()
p.add_argument("--data", default=MCSA_PATH)
p.add_argument("--dedupe", action="store_true",
               help="(source, pdb_basename) 중복 제거. 물리 구조가 평균 2.1회 중복됨")
p.add_argument("--folds", type=int, default=5)
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--groups_per_step", type=int, default=8)
p.add_argument("--negs_per_group", type=int, default=2000)
p.add_argument("--lr", type=float, default=3e-3)
p.add_argument("--lr_linear", type=float, default=3e-2,
               help="선형모델은 수렴이 느려 더 큰 lr 이 필요")
p.add_argument("--patience", type=int, default=8)
p.add_argument("--lam_hinge", type=float, default=1.0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--max_rows", type=int, default=0, help="디버그용 상한 (0=전체)")
args = p.parse_args()


def log(m):
    print(m, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(m + "\n")


# ============================ 그룹 벡터 연산 ============================
def group_stats(qc, s, nq):
    cnt = np.bincount(qc, minlength=nq).astype(np.float64)
    cnt[cnt == 0] = 1.0
    mean = np.bincount(qc, weights=s, minlength=nq) / cnt
    var = np.bincount(qc, weights=s * s, minlength=nq) / cnt - mean ** 2
    return mean, np.sqrt(np.maximum(var, 0.0)) + 1e-9, cnt


def group_z(qc, s, nq):
    m, sd, _ = group_stats(qc, s, nq)
    return (s - m[qc]) / sd[qc]


def group_max(qc, s, nq):
    out = np.full(nq, -np.inf)
    np.maximum.at(out, qc, s)
    return out


def group_pct(qc, s, nq):
    order = np.lexsort((s, qc))
    q_s = qc[order]
    starts = np.flatnonzero(np.r_[True, q_s[1:] != q_s[:-1]])
    counts = np.diff(np.r_[starts, len(order)])
    within = np.arange(len(order)) - np.repeat(starts, counts)
    pct = np.empty(len(order))
    pct[order] = within / np.maximum(np.repeat(counts, counts) - 1, 1)
    return pct


# ============================ 지표 ============================
def map_vectorized(qc, y, s):
    """sklearn average_precision_score 와 동일한 동점 처리. (06 에서 1e-16 까지 검증됨)"""
    nq = qc.max() + 1
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


def first_fp(qc, y, s, tie="pessimistic", rng=None):
    s = np.where(np.isfinite(s), s, -np.inf).astype(np.float64)
    key = (y.astype(np.float64) if tie == "pessimistic"
           else -y.astype(np.float64) if tie == "optimistic" else rng.random(len(y)))
    order = np.lexsort((key, -s, qc))
    q_s, y_s = qc[order], y[order].astype(np.int64)
    starts = np.flatnonzero(np.r_[True, q_s[1:] != q_s[:-1]])
    counts = np.diff(np.r_[starts, len(q_s)])
    gidx = np.repeat(np.arange(len(starts)), counts)
    rank = np.arange(len(q_s)) - np.repeat(starts, counts)
    n_pos = np.add.reduceat(y_s, starts)
    ffp = counts.astype(np.int64).copy()
    neg = y_s == 0
    if neg.any():
        np.minimum.at(ffp, gidx[neg], rank[neg])
    keep = n_pos > 0
    return float((np.minimum(ffp[keep], n_pos[keep]) / n_pos[keep]).mean())


def lex_score(qc, primary, secondary):
    """1순위 primary, 2순위 secondary. 동점이 원천적으로 없다."""
    order = np.lexsort((-secondary, -primary, qc))
    q_s = qc[order]
    starts = np.flatnonzero(np.r_[True, q_s[1:] != q_s[:-1]])
    counts = np.diff(np.r_[starts, len(order)])
    within = np.arange(len(order)) - np.repeat(starts, counts)
    out = np.empty(len(order))
    out[order] = -within.astype(np.float64)
    return out


# ============================ 로드 ============================
log("=" * 80)
log(f"[load] {args.data}")
t0 = time.time()
df = pd.read_csv(args.data, sep="\t", engine="c",
                 nrows=(args.max_rows or None), dtype={"query_num": str, "source": str,
                                                       "target_id": str})
df.columns = [c.strip().replace("\r", "") for c in df.columns]
for c in ("query_num", "source", "target_id"):
    df[c] = df[c].str.replace("\r", "", regex=False)
log(f"[load] rows={len(df)}  ({time.time()-t0:.1f}s)")

if args.dedupe:
    n0 = len(df)
    df["_pdb"] = df["target_id"].str.rsplit("/", n=1).str[-1]
    df = (df.sort_values("label_raw", ascending=False)
            .drop_duplicates(["source", "_pdb"], keep="first")
            .drop(columns=["_pdb"]).reset_index(drop=True))
    log(f"[dedupe] {n0} -> {len(df)}  ({100*(1-len(df)/n0):.1f}% 제거)")

df = df[df["idf"].notna() & df["tm_score"].notna() & (df["L"] > 1) & (df["L"] <= 32)]
df = df.reset_index(drop=True)
if "rmsd" not in df.columns:
    df["rmsd"] = 999.0
df["rmsd"] = df["rmsd"].fillna(999.0)

y = np.where(df["label_raw"].to_numpy() == 0, 0, 1).astype(np.int8)
qc, uq = pd.factorize(df["source"].to_numpy(), sort=False)
qc = qc.astype(np.int64)
nq = len(uq)
log(f"[data] rows={len(df)}  groups={nq}  positives={int(y.sum())} ({100*y.mean():.3f}%)")

# ============================ feature ============================
idf = df["idf"].to_numpy(np.float64)
rmsd = df["rmsd"].to_numpy(np.float64)
tm = df["tm_score"].to_numpy(np.float64)
Lm = df["L"].to_numpy(np.float64)                       # 매칭 크기
Lq = df["query_num"].str.extract(r"length_(\d+)")[0].astype(float).to_numpy()
if np.isnan(Lq).any():
    log(f"[warn] query_num 에서 length_N 파싱 실패 {int(np.isnan(Lq).sum())}건 -> L 로 대체")
    Lq = np.where(np.isnan(Lq), Lm, Lq)

_, _, cnt = group_stats(qc, idf, nq)
gmax = group_max(qc, idf, nq)

FEATS = {
    "z_idf": group_z(qc, idf, nq),
    "z_rmsd": group_z(qc, -rmsd, nq),          # 낮을수록 좋으므로 부호 반전
    "z_tm": group_z(qc, tm, nq),
    "coverage": Lm / np.maximum(Lq, 1.0),      # 매칭 크기 / query 길이
    "idf_pct": group_pct(qc, idf, nq),
    "gap_to_top": (gmax[qc] - idf) / (gmax[qc] + 1e-9),
    "log_nhits": np.log1p(cnt[qc]),            # group 내 상수 (선형모델엔 무효, MLP 는 상호작용 가능)
}
FEAT_NAMES = list(FEATS)
X = np.stack([FEATS[k] for k in FEAT_NAMES], 1).astype(np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
log(f"[feat] {X.shape[1]} features: {FEAT_NAMES}")
del FEATS
gc.collect()

# ============================ 모델 ============================
class LinearScorer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 1)

    def forward(self, x):
        return self.lin(x).squeeze(-1)


class MLPScorer(nn.Module):
    def __init__(self, d, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(),
            nn.Linear(h, h // 2), nn.LayerNorm(h // 2), nn.ReLU(),
            nn.Linear(h // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def listwise_loss(s, yb, lam):
    """group 하나에 대한 loss. InfoNCE(top 가중) + max-negative hinge(first-FP 직결)."""
    pos, neg = s[yb == 1], s[yb == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    lse_neg = torch.logsumexp(neg, 0)
    l_nce = (torch.logaddexp(pos, lse_neg.expand_as(pos)) - pos).mean()
    l_ffp = F.softplus(lse_neg - pos).mean()
    return l_nce + lam * l_ffp


# ============================ 학습 / 평가 ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(args.seed)

# group -> 행 인덱스 (한 번만 만들고 재사용)
order = np.argsort(qc, kind="stable")
gstart = np.searchsorted(qc[order], np.arange(nq))
gend = np.searchsorted(qc[order], np.arange(nq), side="right")
gidx_rows = [order[gstart[g]:gend[g]] for g in range(nq)]
gpos = [r[y[r] == 1] for r in gidx_rows]
gneg = [r[y[r] == 0] for r in gidx_rows]
usable = np.array([len(gpos[g]) > 0 and len(gneg[g]) > 0 for g in range(nq)])
log(f"[data] usable groups (pos&neg 모두 존재): {int(usable.sum())}/{nq}")


def train_one(model, tr_groups, va_groups, tag):
    model = model.to(device)
    lr = args.lr_linear if tag.startswith('A') else args.lr
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    Xva, yva, qva = subset(va_groups)
    best, best_state, bad, first_loss = -1.0, None, 0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(tr_groups)
        tot, nb = 0.0, 0
        for i in range(0, len(perm), args.groups_per_step):
            chunk = perm[i:i + args.groups_per_step]
            losses = []
            for g in chunk:
                pr, ng = gpos[g], gneg[g]
                if len(ng) > args.negs_per_group:
                    ng = rng.choice(ng, args.negs_per_group, replace=False)
                rows = np.concatenate([pr, ng])
                xb = torch.from_numpy(X_STD[rows]).to(device)
                yb = torch.from_numpy(y[rows].astype(np.int64)).to(device)
                l = listwise_loss(model(xb), yb, args.lam_hinge)
                if l is not None:
                    losses.append(l)
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        ep_loss = tot / max(nb, 1)
        if first_loss is None: first_loss = ep_loss
        s = predict(model, Xva)
        m = map_vectorized(qva, yva, s)
        if m > best:
            best, best_state, bad = m, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if ep % 5 == 0 or ep == 1:
            log(f"    [{tag}] ep{ep:02d} loss={ep_loss:.4f} val_mAP={m:.4f} (best {best:.4f})")
        if bad >= args.patience:
            log(f"    [{tag}] early stop @ep{ep} (patience {args.patience})")
            break
    log(f"    [{tag}] converge check: loss {first_loss:.4f} -> {ep_loss:.4f} "
        f"({100*(1-ep_loss/max(first_loss,1e-9)):.1f}% 감소), best val_mAP={best:.4f}")
    model.load_state_dict(best_state)
    return model


def subset(groups):
    rows = np.concatenate([gidx_rows[g] for g in groups])
    q_local = pd.factorize(qc[rows], sort=False)[0].astype(np.int64)
    return X_STD[rows], y[rows], q_local


def predict(model, Xs, bs=1_000_000):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xs), bs):
            out.append(model(torch.from_numpy(Xs[i:i + bs]).to(device))
                       .float().cpu().numpy().astype(np.float64))
    return np.concatenate(out)


# --------- 5-fold CV (group 단위) ---------
ug = np.flatnonzero(usable)
rng.shuffle(ug)
folds = np.array_split(ug, args.folds)
results = {}

for fi, te in enumerate(folds):
    tr_all = np.concatenate([folds[j] for j in range(args.folds) if j != fi])
    rng.shuffle(tr_all)
    ncut = max(int(0.1 * len(tr_all)), 1)
    va, tr = tr_all[:ncut], tr_all[ncut:]
    log(f"\n--- fold {fi+1}/{args.folds}  train={len(tr)} val={len(va)} test={len(te)} ---")

    # feature 표준화: train fold 통계만 사용 (선형모델 수렴에 필수, 누수 방지)
    rows_tr = np.concatenate([gidx_rows[g] for g in tr])
    _mu = X[rows_tr].mean(0, keepdims=True)
    _sd = X[rows_tr].std(0, keepdims=True) + 1e-6
    globals()['X_STD'] = ((X - _mu) / _sd).astype(np.float32)

    rows_te = np.concatenate([gidx_rows[g] for g in te])
    q_te = pd.factorize(qc[rows_te], sort=False)[0].astype(np.int64)
    y_te, idf_te, rmsd_te = y[rows_te], idf[rows_te], rmsd[rows_te]

    # 기준선 (동일 fold)
    fold_res = {
        "Baseline_IDF": idf_te,
        "IDF_then_RMSD": lex_score(q_te, idf_te, -rmsd_te),
    }

    for tag, mk in (("A_Logistic", lambda: LinearScorer(X.shape[1])),
                    ("B_MLP", lambda: MLPScorer(X.shape[1]))):
        torch.manual_seed(args.seed + fi)
        model = train_one(mk(), tr, va, tag)
        s = predict(model, X_STD[rows_te])
        fold_res[tag] = s
        # idf 를 하한으로 보장하는 배포 형태
        u = np.unique(idf_te)
        if len(u) > 1:
            eps = float(np.diff(u).min()) / 4.0
            if eps > 1e-9:
                s01 = np.argsort(np.argsort(s)) / max(len(s) - 1, 1)
                fold_res[f"IDF_eps_{tag}"] = idf_te + eps * s01
        if tag == "A_Logistic":
            w = model.lin.weight.detach().cpu().numpy().ravel()
            log("    [A_Logistic] weights: " +
                "  ".join(f"{n}={v:+.3f}" for n, v in zip(FEAT_NAMES, w)))

    for k, s in fold_res.items():
        results.setdefault(k, []).append(
            (map_vectorized(q_te, y_te, s), first_fp(q_te, y_te, s)))
    gc.collect()

# --------- 리포트 ---------
log("\n" + "=" * 80)
log(f"{'method':<22} {'mAP mean±std':>20} {'first-FP mean±std':>22}")
log("-" * 80)
rows_out = []
for k, v in results.items():
    a = np.array(v)
    log(f"{k:<22} {a[:,0].mean():>12.4f} ±{a[:,0].std():.4f} "
        f"{a[:,1].mean():>14.4f} ±{a[:,1].std():.4f}")
    rows_out.append(dict(method=k, mAP=a[:, 0].mean(), mAP_std=a[:, 0].std(),
                         ffp=a[:, 1].mean(), ffp_std=a[:, 1].std()))
out_csv = os.path.join(OUT_DIR, "listwise_ab_results.csv")
pd.DataFrame(rows_out).to_csv(out_csv, index=False)
log(f"\nsaved -> {out_csv}")

# A vs B 판정 (fold 별 paired 비교)
if "A_Logistic" in results and "B_MLP" in results:
    a, b = np.array(results["A_Logistic"]), np.array(results["B_MLP"])
    d = b[:, 0] - a[:, 0]
    log(f"\n[A/B] fold별 mAP 차이 (B_MLP - A_Logistic): "
        f"{np.round(d,4).tolist()}")
    log(f"[A/B] mean={d.mean():+.4f}  std={d.std():.4f}  B가 이긴 fold={int((d>0).sum())}/{len(d)}")
    if abs(d.mean()) < d.std():
        log("[A/B] -> 차이가 fold 간 분산보다 작다. 단순한 A 를 택하는 것이 타당.")
    elif d.mean() > 0:
        log("[A/B] -> B(MLP) 우세. 비선형이 실제로 기여함.")
    else:
        log("[A/B] -> A(Logistic) 우세. MLP 는 이 feature 에서 과적합.")
log("=" * 80)