"""
06_blend_sweep.py

재학습 없이, 현재 NN logit이 IDF 대비 'query 내부'에서 정보를 더 갖고 있는지 판정한다.

비교 대상
  - Baseline_IDF        : idf 단독 (현재 mAP 0.5395 / first-FP 0.4264 재현되어야 함)
  - NN_alone            : nn logit 단독 (현재 mAP 0.4857 / first-FP 0.3756 재현되어야 함)
  - IDF_then_NN         : 1순위 idf, 2순위 nn logit (사전식). IDF 순서를 절대 바꾸지 않으므로
                          IDF 성능이 수학적 하한. 동점 블록에서만 NN이 개입한다.
  - IDF_then_TM / RMSD  : 동점을 tm_score / -rmsd 로 깨는 대조군 (NN이 이들보다 나은지 확인)
  - Blend(alpha)        : alpha*z_nn + (1-alpha)*z_idf  (query별 정규화 후 선형결합)

tie 규칙은 pessimistic / optimistic / random 세 가지를 모두 보고한다.
random 이 공정한 값이며, 사전식 방법들은 동점이 없어 셋이 일치해야 정상이다.
"""

import os
import gc
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

# ---------------- 설정 ----------------
RESULT_DIR = "result_benchmark_v2"
SCORE_DIR = os.path.join(RESULT_DIR, "folddisco_results_scores_test")
INPUT_DATA_PATH = os.path.join(RESULT_DIR, "folddisco_results_analyses/data_test_scope40.txt")
MODEL_SAVE_PATH = os.path.join(SCORE_DIR, "best_mlp_model.pt")
SCALER_SAVE_PATH = os.path.join(SCORE_DIR, "scaler_stats.npz")
LOG_FILE = os.path.join(SCORE_DIR, "blend_sweep_log.txt")
QUERY_KEY = "target_id"

parser = argparse.ArgumentParser()
parser.add_argument("--norm", choices=["z", "rank"], default="z",
                    help="blend 시 query별 정규화 방식. rank=percentile(이상치에 강건)")
parser.add_argument("--alphas", type=str, default="0,0.1,0.2,0.3,0.4,0.5,0.7,1.0")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ---------------- query 단위 벡터 연산 (O(n), groupby 미사용) ----------------
def group_z(qc, s, nq):
    """query별 z-score. query 내 monotone 이므로 단독 사용 시 순위 불변."""
    cnt = np.bincount(qc, minlength=nq).astype(np.float64)
    cnt[cnt == 0] = 1.0
    mean = np.bincount(qc, weights=s, minlength=nq) / cnt
    var = np.bincount(qc, weights=s * s, minlength=nq) / cnt - mean ** 2
    sd = np.sqrt(np.maximum(var, 0.0)) + 1e-9
    return (s - mean[qc]) / sd[qc]


def group_rank_pct(qc, s, nq):
    """query별 percentile rank (0~1). 동점은 평균 순위가 아니라 순차 부여."""
    order = np.lexsort((s, qc))
    starts = np.flatnonzero(np.r_[True, qc[order][1:] != qc[order][:-1]])
    counts = np.diff(np.r_[starts, len(order)])
    within = np.arange(len(order)) - np.repeat(starts, counts)
    pct = np.empty(len(order), np.float64)
    pct[order] = within / np.maximum(np.repeat(counts, counts) - 1, 1)
    return pct


def lex_score(qc, primary, secondary, nq):
    """1순위 primary, 2순위 secondary 로 정렬한 뒤 query 내 순위를 점수로 반환.
    순위가 정수로 유일하므로 동점이 원천적으로 없다."""
    order = np.lexsort((-secondary, -primary, qc))
    q_s = qc[order]
    starts = np.flatnonzero(np.r_[True, q_s[1:] != q_s[:-1]])
    counts = np.diff(np.r_[starts, len(order)])
    within = np.arange(len(order)) - np.repeat(starts, counts)
    out = np.empty(len(order), np.float64)
    out[order] = -within.astype(np.float64)   # 순위가 낮을수록(앞설수록) 높은 점수
    return out


# ---------------- 지표 ----------------
def map_vectorized(qc, y, s, nq):
    """query별 AP의 평균. sklearn average_precision_score 와 동일한 동점 처리
    (같은 점수를 하나의 threshold 블록으로 묶음). 양·음성 모두 있는 query만 사용."""
    s = np.where(np.isfinite(s), s, -np.inf)
    order = np.lexsort((-s, qc))
    q_s, y_s, s_s = qc[order], y[order].astype(np.float64), s[order]

    starts = np.flatnonzero(np.r_[True, q_s[1:] != q_s[:-1]])
    counts = np.diff(np.r_[starts, len(q_s)])
    gidx = np.repeat(np.arange(len(starts)), counts)

    cum_tp = np.cumsum(y_s)
    cum_tp -= np.repeat(np.r_[0.0, cum_tp[starts[1:] - 1]], counts)   # group별 누적 리셋
    cum_n = np.arange(len(q_s)) - np.repeat(starts, counts) + 1.0

    # 각 group 내에서 '점수가 바뀌는 지점(블록 끝)'만 남긴다
    is_last = np.r_[True, q_s[1:] != q_s[:-1]][1:]                    # 다음 원소가 새 group
    is_last = np.r_[is_last, True]
    block_end = is_last | np.r_[s_s[1:] != s_s[:-1], True]

    n_pos = np.add.reduceat(y_s, starts)
    valid = (n_pos > 0) & (n_pos < counts)

    prec = cum_tp / cum_n
    rec = cum_tp / np.maximum(np.repeat(n_pos, counts), 1e-12)

    be = np.flatnonzero(block_end)
    g_be, p_be, r_be = gidx[be], prec[be], rec[be]
    prev_r = np.r_[0.0, r_be[:-1]]
    prev_r[np.r_[True, g_be[1:] != g_be[:-1]]] = 0.0                  # group 첫 블록
    ap = np.bincount(g_be, weights=(r_be - prev_r) * p_be, minlength=len(starts))

    return float(ap[valid].mean()), int(valid.sum())


def first_fp(qc, y, s, nq, tie="pessimistic", rng=None):
    s = np.where(np.isfinite(s), s, -np.inf).astype(np.float64)
    if tie == "pessimistic":
        key = y.astype(np.float64)
    elif tie == "optimistic":
        key = -y.astype(np.float64)
    else:
        key = rng.random(len(y))
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


# ---------------- 데이터 로드 ----------------
log("=" * 78)
log(f"Loading {INPUT_DATA_PATH} ...")
df = pd.read_csv(INPUT_DATA_PATH, sep=r"\s+", engine="c")

mask = (df["idf"].notna() & df["tm_score"].notna() & (df["L"] > 1) & (df["L"] <= 32))
df = df[mask].copy().reset_index(drop=True)
if "rmsd" not in df.columns:
    df["rmsd"] = 999.0
y = np.where(df["label_raw"] == 0, 0, 1).astype(np.int8)

qc, uq = pd.factorize(df[QUERY_KEY].to_numpy(), sort=False)
qc = qc.astype(np.int64)
nq = len(uq)
log(f"rows={len(df)}  queries={nq}  positives={int(y.sum())}")

# ---------------- NN 추론 (05와 동일, autocast 없음 / float64) ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleMLP(nn.Module):
    def __init__(self, in_dim=4, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


X = df[["L", "idf", "rmsd", "tm_score"]].values.astype(np.float32)
sc = np.load(SCALER_SAVE_PATH)
X = (X - sc["mu"]) / sc["sd"]

model = SimpleMLP().to(device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval()

chunks = []
with torch.no_grad():
    for i in range(0, len(X), 262144):
        bx = torch.tensor(X[i:i + 262144], dtype=torch.float32).to(device)
        chunks.append(model(bx).float().cpu().numpy().astype(np.float64).reshape(-1))
nn_logit = np.concatenate(chunks)
del chunks, X
gc.collect()

idf = df["idf"].to_numpy(np.float64)
tm = df["tm_score"].to_numpy(np.float64)
neg_rmsd = -df["rmsd"].to_numpy(np.float64)

# ---------------- 후보 점수 구성 ----------------
norm = group_z if args.norm == "z" else group_rank_pct
z_nn, z_idf = norm(qc, nn_logit, nq), norm(qc, idf, nq)

methods = {
    "Baseline_IDF": idf,
    "NN_alone": nn_logit,
    "IDF_then_NN": lex_score(qc, idf, nn_logit, nq),     # ★ IDF 성능이 하한으로 보장
    "IDF_then_TM": lex_score(qc, idf, tm, nq),           # 대조군
    "IDF_then_RMSD": lex_score(qc, idf, neg_rmsd, nq),   # 대조군
}

# IDF_then_* 는 query 내 '순위'라서 global AP 가 의미 없다(query 간 비교 불가).
# 아래는 idf 절대 스케일을 보존하면서 동점만 깨는 형태 -> global AP 도 유효.
_u = np.unique(idf)
_min_gap = float(np.diff(_u).min()) if len(_u) > 1 else 1.0
log(f"[eps] distinct idf={len(_u)}  min gap={_min_gap:.6g}")
if _min_gap > 1e-9:
    _eps = _min_gap / 4.0
    _nn01 = np.argsort(np.argsort(nn_logit)) / max(len(nn_logit) - 1, 1)   # 전역 0~1
    methods["IDF_eps_NN"] = idf + _eps * _nn01          # idf 순서 완전 보존 + 동점 해소
else:
    log("[eps] min gap 이 너무 작아 IDF_eps_NN 생략 (IDF_then_NN 으로 판단할 것)")

for a in [float(v) for v in args.alphas.split(",")]:
    methods[f"Blend_a{a:.2f}"] = a * z_nn + (1 - a) * z_idf

# ---------------- 평가 ----------------
rng = np.random.default_rng(args.seed)
log("")
log(f"{'method':<18} {'mAP':>8} {'globalAP':>9} {'ffp_pess':>9} {'ffp_rand':>9} {'ffp_opt':>8} {'gap':>7}")
log("-" * 78)

rows = []
for name, s in methods.items():
    m, nvalid = map_vectorized(qc, y, s, nq)
    gap_ap = average_precision_score(y, s)
    fp_p = first_fp(qc, y, s, nq, "pessimistic")
    fp_r = first_fp(qc, y, s, nq, "random", rng)
    fp_o = first_fp(qc, y, s, nq, "optimistic")
    log(f"{name:<18} {m:>8.4f} {gap_ap:>9.4f} {fp_p:>9.4f} {fp_r:>9.4f} {fp_o:>8.4f} {fp_o - fp_p:>7.4f}")
    rows.append(dict(method=name, mAP=m, global_AP=gap_ap,
                     ffp_pessimistic=fp_p, ffp_random=fp_r, ffp_optimistic=fp_o))
    gc.collect()

out_csv = os.path.join(SCORE_DIR, "blend_sweep_results.csv")
pd.DataFrame(rows).to_csv(out_csv, index=False)
log("")
log(f"saved -> {out_csv}")

# ---------------- query별 승패 귀인 ----------------
log("")
log("[attribution] NN_alone 이 IDF 에게 지는 query 의 특성")


def per_query_ap(s):
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
    prec, rec = cum_tp / cum_n, cum_tp / np.maximum(np.repeat(n_pos, counts), 1e-12)
    g_be, p_be, r_be = gidx[be], prec[be], rec[be]
    prev = np.r_[0.0, r_be[:-1]]
    prev[np.r_[True, g_be[1:] != g_be[:-1]]] = 0.0
    ap = np.bincount(g_be, weights=(r_be - prev) * p_be, minlength=len(starts))
    return ap, n_pos, counts


ap_nn, n_pos, n_hits = per_query_ap(nn_logit)
ap_idf, _, _ = per_query_ap(idf)
valid = (n_pos > 0) & (n_pos < n_hits)
d = ap_nn[valid] - ap_idf[valid]

# query motif 길이는 L(=매칭 크기)이 아니라 query_num 의 length_N 에 들어있다
Lq = np.zeros(nq)
if "query_num" in df.columns:
    _ql = df["query_num"].astype(str).str.extract(r"length_(\d+)")[0].astype(float).to_numpy()
    Lq[qc] = np.nan_to_num(_ql, nan=0.0)
    _label = "L_query"
else:
    np.maximum.at(Lq, qc, df["L"].to_numpy(np.float64))
    _label = "L_max(fallback)"
Lq, nh, npos_v = Lq[valid], n_hits[valid], n_pos[valid]

log(f"  NN 승: {(d > 0).mean():.3f}   무승부: {(d == 0).mean():.3f}   IDF 승: {(d < 0).mean():.3f}")
for lo, hi in [(0, 11), (11, 21), (21, 31), (31, 1000)]:
    m = (Lq >= lo) & (Lq < hi)
    if m.sum():
        log(f"  {_label}[{lo:>3},{hi:>4}) n={m.sum():>6}  mean dAP={d[m].mean():+.4f}  "
            f"NN win rate={(d[m] > 0).mean():.3f}")
for q_lo, q_hi, lab in [(0, 25, "hits Q1"), (25, 50, "Q2"), (50, 75, "Q3"), (75, 101, "Q4")]:
    lo_v, hi_v = np.percentile(nh, q_lo), np.percentile(nh, min(q_hi, 100))
    m = (nh >= lo_v) & (nh <= hi_v)
    if m.sum():
        log(f"  {lab:<8} n={m.sum():>6}  mean dAP={d[m].mean():+.4f}  "
            f"NN win rate={(d[m] > 0).mean():.3f}")
log("=" * 78)