import argparse
import os
import gc
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from folddisco_features import (FEATURE_NAMES, normalize_columns,
                                build_features, group_size_guard, per_query_z)
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score
)

# 1) 외부 스타일시트 적용 및 배경/그리드 설정
plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.facecolor'] = 'white'

# ---------- 0) CLI Argument Parsing ----------
parser = argparse.ArgumentParser(description="Evaluate Significance Score model and Baseline performance on scope40 test set.")
parser.add_argument(
    "--optimal_threshold", "-t", 
    type=float, 
    default=None, 
    help="Custom threshold for Significance Score model. If not specified, automatically determined via Max F1-Score."
)
parser.add_argument("--all_variants", action="store_true",
    help="ablation 용 변형(NN_raw_logit, NN_prob)도 함께 평가. 기본은 Significance Score 하나.")
parser.add_argument("--plot_models", default="",
    help="combined plot(ROC/PR/first-FP/FPR)에 그릴 모델만 쉼표로 지정. 비우면 전체.")
parser.add_argument("--main_figure", action="store_true",
    help="main figure 용. baseline 전부 + IDF_then_RMSD + Significance Score 를 그리고 ablation 변형만 제외.")
parser.add_argument("--score_dir", default="folddisco_results_scores_final2",
    help="점수/로그 디렉토리")
parser.add_argument("--plot_dir", default="analyses_plot_comparison_final2",
    help="plot 저장 디렉토리")
parser.add_argument("--model_in", default="",
    help="모델 .pt 경로")
parser.add_argument(
    "--threshold_scale",
    choices=["auto", "prob", "logit"],
    default="auto"
)
args = parser.parse_args()

RESULT_DIR = "result_benchmark_v2"
SCORE_DIR = os.path.join(RESULT_DIR, args.score_dir)
INPUT_DATA_PATH = os.path.join(RESULT_DIR, "folddisco_results_analyses/data_test_scope40.txt")
MODEL_SAVE_PATH = args.model_in or os.path.join(SCORE_DIR, "best_mlp_model.pt")

if args.plot_dir:
    PNG_DIR = os.path.join(RESULT_DIR, args.plot_dir)
elif args.optimal_threshold is not None:
    PNG_DIR = os.path.join(RESULT_DIR, f"analyses_plot_comparison_{args.optimal_threshold:g}")
else:
    PNG_DIR = os.path.join(RESULT_DIR, "analyses_plot_comparison_test")

os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(SCORE_DIR, exist_ok=True)

LOG_FILE = os.path.join(SCORE_DIR, "analysis_log.txt")
RESULTS_FILE = os.path.join(SCORE_DIR, "evaluation_results_summary.txt")

# 플롯 경로 정의
AUC_ROC_FILE = os.path.join(PNG_DIR, "combined_roc_curve.png")
AUC_PR_FILE = os.path.join(PNG_DIR, "combined_precision_recall_curve.png")
FIRST_FP_PLOT_FILE = os.path.join(PNG_DIR, "combined_first_fp_plot.png")
FPR_BAR_PLOT_FILE = os.path.join(PNG_DIR, "combined_fpr_bar_plot.png")
FEATURE_DIST_FILE = os.path.join(PNG_DIR, "input_features_distribution_raw.png")

def log_and_print(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

log_and_print(f"Plots and figures will be saved to: {PNG_DIR}")


# ---------- [TIE-FIX] 수치 안정 sigmoid / logit ----------
def _stable_sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _logit(p):
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    return float(np.log(p / (1.0 - p)))

QUERY_KEY = 'target_id'


def compute_map(df, score_col, y, log):
    sc = df[score_col].to_numpy(dtype=np.float64)
    finite = np.isfinite(sc)
    n_dropped = int((~finite).sum())

    sub = pd.DataFrame({
        "q": df[QUERY_KEY].to_numpy()[finite],
        "y": np.asarray(y)[finite],
        "s": sc[finite],
    })

    aps, n_single = [], 0
    for _, g in sub.groupby("q", sort=False):
        yv = g["y"].to_numpy()
        if yv.min() == yv.max():
            n_single += 1
            continue
        aps.append(average_precision_score(yv, g["s"].to_numpy()))

    map_val = float(np.mean(aps)) if aps else float("nan")
    log(f"mAP over {len(aps)} queries (single-class skipped: {n_single}, non-finite score dropped: {n_dropped})")
    return map_val


def first_fp_fractions(qid_codes, y, score, tie="pessimistic", return_stats=False):
    qid_codes = np.asarray(qid_codes, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(score, dtype=np.float64)
    s = np.where(np.isfinite(s), s, -np.inf)

    tie_key = y if tie == "pessimistic" else -y
    order = np.lexsort((tie_key, -s, qid_codes))
    q_s, y_s = qid_codes[order], y[order]

    starts = np.flatnonzero(np.r_[True, q_s[1:] != q_s[:-1]])
    counts = np.diff(np.r_[starts, len(q_s)])
    gidx = np.repeat(np.arange(len(starts)), counts)
    rank = np.arange(len(q_s)) - np.repeat(starts, counts)

    n_pos = np.add.reduceat(y_s, starts)

    first_fp = counts.astype(np.int64).copy()
    neg = y_s == 0
    if neg.any():
        np.minimum.at(first_fp, gidx[neg], rank[neg])

    keep = n_pos > 0
    frac = np.minimum(first_fp[keep], n_pos[keep]) / n_pos[keep]

    if not return_stats:
        return frac

    s_s = s[order]
    top = np.repeat(s_s[starts], counts)
    at_top = s_s == top
    n_top = np.bincount(gidx[at_top], minlength=len(starts))
    pos_top = np.bincount(gidx[at_top], weights=y_s[at_top].astype(np.float64), minlength=len(starts))
    mixed = (n_top > 1) & (pos_top > 0) & (pos_top < n_top)
    stats = {
        "tied_top_frac": float((n_top[keep] > 1).mean()),
        "mixed_top_frac": float(mixed[keep].mean()),
        "mean_top_block": float(n_top[keep].mean()),
    }
    return frac, stats


# ---------- [스타일 헬퍼] ----------
OTHER_LINESTYLES = ['--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1))]
_style_counter = 0

def get_model_style(raw_name):
    global _style_counter
    lower_name = str(raw_name).lower()
    
    if any(k in lower_name for k in ['significance', 'folddisco-nn', 'folddisco_nn', '_nn']):
        display_name = 'Significance Score'
    else:
        display_name = raw_name.replace('Baseline_', '').replace('score_', '')

    if display_name == 'Significance Score':
        color = '#008080'         # 진한 청록(Teal)
        light_color = '#20b2aa'   # Bar plot용 밝은 Teal
        lw = 3.2
        ls = '-'
        zorder = 5
    else:
        color = '#7f8c8d'         # 차분한 회색 계열
        light_color = '#bdc3c7'   # Bar plot용 연회색
        lw = 1.8
        ls = OTHER_LINESTYLES[_style_counter % len(OTHER_LINESTYLES)]
        _style_counter += 1
        zorder = 2
    
    return display_name, color, light_color, lw, ls, zorder


# ---------- 1) 데이터 로드 및 NN 추론 ----------
log_and_print(f"Loading scope40 test dataset from {INPUT_DATA_PATH}...")
df_raw = pd.read_csv(INPUT_DATA_PATH, sep=r'\s+', engine='c')

mask = (
    df_raw['idf'].notna() & 
    df_raw['tm_score'].notna() & 
    (df_raw['L'] > 1) & 
    (df_raw['L'] <= 32)
)
df_raw = df_raw[mask].copy().reset_index(drop=True)

if 'rmsd' not in df_raw.columns: df_raw['rmsd'] = 999.0
if 'node_count' not in df_raw.columns: df_raw['node_count'] = df_raw['L']
df_raw['y'] = np.where(df_raw['label_raw'] == 0, 0, 1).astype(np.int8)

assert QUERY_KEY in df_raw.columns, f"{QUERY_KEY} column not found: {list(df_raw.columns)}"
log_and_print(f"[sanity] unique {QUERY_KEY} (query)     : {df_raw[QUERY_KEY].nunique()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleMLP(nn.Module):
    def __init__(self, in_dim, hidden=256, skip="none"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1))
        self.skip = nn.Linear(in_dim, 1) if skip == "linear" else None

    def forward(self, x):
        s = self.net(x).squeeze(-1)
        if self.skip is not None:
            s = s + self.skip(x).squeeze(-1)
        return s


class LinearScorer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.lin(x).squeeze(-1)


df_raw = normalize_columns(df_raw)
ckpt = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False)
_names = [str(v) for v in ckpt["feature_names"]]
X_test, _qc_feat, _nq_feat = build_features(df_raw, QUERY_KEY, _names)

X_test_scaled = ((X_test - ckpt["mu"]) / ckpt["sd"]).astype(np.float32)
del X_test
gc.collect()

model = (SimpleMLP(ckpt["in_dim"], ckpt.get("hidden", 256), ckpt.get("skip", "none"))
         if ckpt["arch"] == "mlp" else LinearScorer(ckpt["in_dim"])).to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()

logits_list = []
batch_size = 262144

with torch.no_grad():
    for i in range(0, len(X_test_scaled), batch_size):
        bx = torch.tensor(X_test_scaled[i:i+batch_size], dtype=torch.float32).to(device)
        batch_logits = model(bx)
        logits_list.append(batch_logits.float().cpu().numpy().astype(np.float64).reshape(-1))

# 1) score_logit_NN (Raw Logit)
df_raw['score_logit_NN'] = np.concatenate(logits_list).astype(np.float64)
del logits_list
gc.collect()

# 2) score_NN: 방안 B (온도 조절 로지스틱 스케일링: z=20 -> 0.95, z=-8 -> 0.05)
# 엄격한 단조 증가 함수 -> Precision, Recall, mAP 100% 완전 보존
_lg = df_raw['score_logit_NN'].to_numpy(np.float64)
SCALE_MU = 6.0
SCALE_T = 28.0 / (2.0 * np.log(19.0))  # ≈ 4.75474
df_raw['score_NN'] = _stable_sigmoid((_lg - SCALE_MU) / SCALE_T)

# 3) score_NN_z: Per-query Z-score
_qz = df_raw[QUERY_KEY].factorize()[0]
_nz = _qz.max() + 1
_c = np.bincount(_qz, minlength=_nz).astype(np.float64); _c[_c == 0] = 1.0
_mn = np.bincount(_qz, weights=_lg, minlength=_nz) / _c
_sd = np.sqrt(np.maximum(np.bincount(_qz, weights=_lg * _lg, minlength=_nz) / _c - _mn ** 2, 0)) + 1e-9
df_raw['score_NN_z'] = (_lg - _mn[_qz]) / _sd[_qz]
del _lg, _c, _mn, _sd
gc.collect()

# Baseline 스코어 구성
df_raw['score_IDF'] = df_raw['idf']
df_raw['score_TM'] = df_raw['tm_score']
df_raw['score_RMSD'] = -df_raw['rmsd']

_o = np.lexsort((df_raw['rmsd'].to_numpy(), -df_raw['idf'].to_numpy(),
                 df_raw[QUERY_KEY].factorize()[0]))
_q = df_raw[QUERY_KEY].factorize()[0][_o]
_st = np.flatnonzero(np.r_[True, _q[1:] != _q[:-1]])
_ct = np.diff(np.r_[_st, len(_o)])
_rk = np.empty(len(_o))
_rk[_o] = -(np.arange(len(_o)) - np.repeat(_st, _ct)).astype(np.float64)
df_raw['score_IDF_then_RMSD'] = _rk
df_raw['score_RMSD_Node'] = np.where(df_raw['rmsd'] <= 2.5, df_raw['node_count'], df_raw['node_count'] - 100000.0)

y_test = df_raw['y'].values
n_true_samples = int(np.sum(y_test))

# 평가 대상 모델 정의
models = {
    'Significance Score': {'score_col': 'score_NN_z', 'type': 'nn'},
    'Baseline_IDF': {'score_col': 'score_IDF', 'type': 'top_n'},
    'Baseline_RMSD': {'score_col': 'score_RMSD', 'type': 'top_n'},
    'Baseline_TM': {'score_col': 'score_TM', 'type': 'top_n'},
    'Baseline_RMSD_Node': {'score_col': 'score_RMSD_Node', 'type': 'top_n'},
    'IDF_then_RMSD': {'score_col': 'score_IDF_then_RMSD', 'type': 'top_n'},
}

MAIN_MODELS = list(models)

if args.all_variants:
    models['NN_raw_logit'] = {'score_col': 'score_logit_NN', 'type': 'nn'}
    models['NN_prob'] = {'score_col': 'score_NN', 'type': 'nn'}

PLOT_SAMPLE_SIZE = 10000000
if len(y_test) > PLOT_SAMPLE_SIZE:
    plot_indices = np.random.choice(len(y_test), PLOT_SAMPLE_SIZE, replace=False)
else:
    plot_indices = np.arange(len(y_test))

y_test_sampled = y_test[plot_indices]

if args.main_figure and not args.plot_models:
    args.plot_models = ",".join(MAIN_MODELS)

_PLOT_SET = None
if args.plot_models:
    raw_plot_list = [m.strip() for m in args.plot_models.split(",") if m.strip()]
    normalized_plot_list = []
    for m in raw_plot_list:
        if any(k in m.lower() for k in ['significance', 'folddisco-nn', 'folddisco_nn', '_nn']):
            normalized_plot_list.append('Significance Score')
        else:
            normalized_plot_list.append(m)
    _PLOT_SET = normalized_plot_list


def _in_plot(name):
    return _PLOT_SET is None or name in _PLOT_SET


# ---------- 플롯 객체 및 FPR 수집용 리스트 ----------
fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
fig_pr, ax_pr = plt.subplots(figsize=(10, 8))
fig_ffp, ax_ffp = plt.subplots(figsize=(11, 8))

summary_lines = []
fpr_bar_data = [] 
_style_counter = 0

# ---------- 모든 모델 루프 ----------
for model_name, info in models.items():
    log_and_print(f"\n--- Evaluating Model: {model_name} ---")
    y_probs = df_raw[info['score_col']].values
    disp_name, p_color, p_light_color, p_lw, p_ls, p_zorder = get_model_style(model_name)

    if info['type'] == 'nn':
        if args.optimal_threshold is not None:
            t_in = args.optimal_threshold
            if args.threshold_scale == "logit" or (args.threshold_scale == "auto" and not (0.0 < t_in < 1.0)):
                optimal_threshold = t_in
            else:
                optimal_threshold = _logit(t_in)
        else:
            sub_idx = np.random.choice(len(y_test), min(5000000, len(y_test)), replace=False)
            precisions, recalls, thresholds_pr = precision_recall_curve(y_test[sub_idx], y_probs[sub_idx])
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            best_idx_pr = np.argmax(f1_scores)
            optimal_threshold = thresholds_pr[best_idx_pr] if best_idx_pr < len(thresholds_pr) else thresholds_pr[-1]
    else:
        sorted_scores = np.sort(y_probs)[::-1]
        optimal_threshold = sorted_scores[n_true_samples - 1] if n_true_samples > 0 and n_true_samples <= len(sorted_scores) else 0.5
        del sorted_scores

    if info['type'] == 'nn':
        NN_THRESHOLD_LOGIT = float(optimal_threshold)

    preds_bool = y_probs >= optimal_threshold
    y_pred_bin = preds_bool.astype(np.int8)

    acc_val = accuracy_score(y_test, y_pred_bin)
    prec_val = precision_score(y_test, y_pred_bin, zero_division=0)
    rec_val = recall_score(y_test, y_pred_bin, zero_division=0)
    f1_val = f1_score(y_test, y_pred_bin, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_probs)
    global_ap = average_precision_score(y_test, y_probs)
    map_val = compute_map(df_raw, info['score_col'], y_test, log_and_print)

    # FPR 계산
    cm = confusion_matrix(y_test, y_pred_bin)
    tn, fp, fn, tp = cm.ravel()
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    summary_lines.append(
        f"[{disp_name}]\nThreshold: {optimal_threshold:.6f}\nFPR: {fpr_val:.6f}\n"
        f"ROC AUC: {roc_auc:.4f}\nGlobal AP: {global_ap:.4f}\nmAP: {map_val:.4f}\n"
        f"Precision: {prec_val:.4f}\nRecall: {rec_val:.4f}\nF1 Score: {f1_val:.4f}\n\n"
    )

    y_probs_sampled = y_probs[plot_indices]

    # 1) ROC Curve 플롯 누적
    fpr_roc, tpr_roc, _ = roc_curve(y_test_sampled, y_probs_sampled)
    if _in_plot(model_name):
        ax_roc.plot(
            fpr_roc, tpr_roc, 
            color=p_color, lw=p_lw, linestyle=p_ls, zorder=p_zorder, 
            label=f'{disp_name} ({roc_auc:.2f})'
        )

    # 2) PR Curve 플롯 누적
    precisions, recalls, _ = precision_recall_curve(y_test_sampled, y_probs_sampled)
    if _in_plot(model_name):
        ax_pr.plot(
            recalls, precisions, 
            color=p_color, lw=p_lw, linestyle=p_ls, zorder=p_zorder, 
            label=f'{disp_name} ({global_ap:.2f})'
        )
        fpr_bar_data.append({
            'name': disp_name,
            'fpr': fpr_val,
            'color': p_color,
            'light_color': p_light_color,
            'ls': p_ls,
            'is_target': (disp_name == 'Significance Score')
        })

    # ---------- [개별 플롯 1] Confusion Matrix ----------
    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f'Confusion Matrix - {disp_name}\n(Cutoff={optimal_threshold:.2f})', fontsize=18, fontweight='bold')
    cbar = fig.colorbar(cax)
    cbar.ax.tick_params(labelsize=14)
    classes = ['Non-homologous', 'Homologous']
    ax.set_xticks([0, 1], labels=classes, fontsize=14)
    ax.set_yticks([0, 1], labels=classes, fontsize=14)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    fontsize=16,
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel('True label', fontsize=16)
    ax.set_xlabel('Predicted label', fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(PNG_DIR, f"confusion_matrix_{disp_name}.png"))
    plt.close(fig)

    # ---------- [개별 플롯 2] TP vs FP vs Threshold ----------
    thresh_vals = np.linspace(y_probs_sampled.min(), y_probs_sampled.max(), 100)
    tps, fps = [], []
    for t in thresh_vals:
        preds = (y_probs_sampled >= t)
        tps.append(np.sum((y_test_sampled == 1) & preds))
        fps.append(np.sum((y_test_sampled == 0) & preds))

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(thresh_vals, fps, color='#e74c3c', lw=2.2, linestyle='--', label='False Positives (FP)')
    ax.plot(thresh_vals, tps, color='#008080', lw=2.5, label='True Positives (TP)')
    ax.axvline(x=optimal_threshold, color='#2c3e50', linestyle=':', lw=2, label=f'Cutoff ({optimal_threshold:.2f})')
    ax.set_xlabel('Threshold', fontsize=18)
    ax.set_ylabel('Count (Log Scale)', fontsize=18)
    ax.set_yscale('log', nonpositive='clip') 
    ax.set_title(f'TP and FP Counts vs. Threshold\n({disp_name})', fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.legend(fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(PNG_DIR, f"tp_fp_vs_threshold_{disp_name}.png"))
    plt.close(fig)

    # ---------- [개별 플롯 3] Score Distribution ----------
    TP_idx = (y_test_sampled == 1) & (y_probs_sampled >= optimal_threshold)
    FP_idx = (y_test_sampled == 0) & (y_probs_sampled >= optimal_threshold)
    FN_idx = (y_test_sampled == 1) & (y_probs_sampled < optimal_threshold)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(y_probs_sampled[FP_idx], bins=50, alpha=0.7, color='#e74c3c', label='False Positive (FP - Alarms)')
    ax.hist(y_probs_sampled[TP_idx], bins=50, alpha=0.8, color='#008080', label='True Positive (TP - Hits)')
    ax.hist(y_probs_sampled[FN_idx], bins=50, alpha=0.5, color='#f39c12', label='False Negative (FN - Misses)')
    ax.axvline(x=optimal_threshold, color='#2c3e50', linestyle='dashed', linewidth=2, label=f'Threshold ({optimal_threshold:.2f})')
    ax.set_xlabel('Significance Score' if info['type'] == 'nn' else 'Score', fontsize=18)
    ax.set_ylabel('Count (Log Scale)', fontsize=18)
    ax.set_yscale('log', nonpositive='clip')
    ax.set_title(f'Distribution of Scores - {disp_name}', fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.legend(loc='upper center', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(PNG_DIR, f"score_distribution_{disp_name}.png"))
    plt.close(fig)

    # ---------- [CSV 출력: score_logit_NN, score_NN, score_NN_z만 보존] ----------
    fp_mask = (y_test == 0) & preds_bool
    fn_mask = (y_test == 1) & (~preds_bool)
    tp_mask = (y_test == 1) & preds_bool

    save_cols = [c for c in df_raw.columns if not c.startswith('_')]
    df_raw.loc[fp_mask, save_cols].sort_values(by=info['score_col'], ascending=False).to_csv(
        os.path.join(SCORE_DIR, f"{disp_name}_false_positives_top.csv"), index=False)
    df_raw.loc[fn_mask, save_cols].sort_values(by=info['score_col'], ascending=True).to_csv(
        os.path.join(SCORE_DIR, f"{disp_name}_false_negatives_top.csv"), index=False)
    df_raw.loc[tp_mask, save_cols].sort_values(by=info['score_col'], ascending=False).to_csv(
        os.path.join(SCORE_DIR, f"{disp_name}_true_positives_top.csv"), index=False)

    del y_probs, preds_bool, y_pred_bin, y_probs_sampled
    gc.collect()

# ==============================================================================
# 1) 공통 ROC Curve 저장
# ==============================================================================
ax_roc.plot([0, 1], [0, 1], color='#bdc3c7', linestyle='--', linewidth=1.5)
ax_roc.set_xlabel('False Positive Rate', fontsize=20)
ax_roc.set_ylabel('True Positive Rate', fontsize=20)
ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve Comparison', fontsize=22, fontweight='bold')
ax_roc.tick_params(axis='both', which='major', labelsize=18)
ax_roc.legend(loc='lower right', fontsize=16)
fig_roc.tight_layout()
fig_roc.savefig(AUC_ROC_FILE)
plt.close(fig_roc)

# ==============================================================================
# 2) 공통 PR Curve 저장
# ==============================================================================
ax_pr.set_xlabel('Recall', fontsize=20)
ax_pr.set_ylabel('Precision', fontsize=20)
ax_pr.set_title('Precision-Recall Curve Comparison', fontsize=22, fontweight='bold')
ax_pr.tick_params(axis='both', which='major', labelsize=18)
ax_pr.legend(loc='upper right', fontsize=18)
fig_pr.tight_layout()
fig_pr.savefig(AUC_PR_FILE)
plt.close(fig_pr)

# ==============================================================================
# 3) First-FP 플롯 생성 및 저장
# ==============================================================================
FFP_TIE = "pessimistic"
_qid = pd.factorize(df_raw[QUERY_KEY].to_numpy(), sort=False)[0]
_style_counter = 0

for model_name, info in models.items():
    sc = df_raw[info['score_col']].to_numpy(dtype=np.float64)
    fr = first_fp_fractions(_qid, y_test, sc, tie=FFP_TIE)
    if len(fr) == 0:
        continue

    fr_sorted = np.sort(fr)[::-1]
    x = np.arange(1, len(fr_sorted) + 1) / len(fr_sorted)
    ffp_auc = float(fr.mean())

    disp_name, p_color, _, p_lw, p_ls, p_zorder = get_model_style(model_name)

    if _in_plot(model_name):
        ax_ffp.plot(
            x, fr_sorted, 
            color=p_color, lw=p_lw, linestyle=p_ls, zorder=p_zorder, 
            label=f'{disp_name} ({ffp_auc:.2f})'
        )
    del fr, sc
    gc.collect()

ax_ffp.set_xlabel('Fraction of queries (sorted, descending)', fontsize=20)
ax_ffp.set_ylabel('Fraction of TPs found until first FP', fontsize=20)
ax_ffp.set_title('First FP Criterion', fontsize=22, fontweight='bold')
ax_ffp.tick_params(axis='both', which='major', labelsize=18)
ax_ffp.legend(loc='upper right', fontsize=18, frameon=True, shadow=True)
fig_ffp.tight_layout()
fig_ffp.savefig(FIRST_FP_PLOT_FILE)
plt.close(fig_ffp)
log_and_print(f"[PLOT] Saved First-FP plot to {FIRST_FP_PLOT_FILE}")

# ==============================================================================
# 4) False Positive Rate (FPR) Bar Plot 생성
# ==============================================================================
fig_bar, ax_bar = plt.subplots(figsize=(10, 7))

names = [item['name'] for item in fpr_bar_data]
fpr_vals = [item['fpr'] * 100 for item in fpr_bar_data]
colors = [item['color'] if item['is_target'] else item['light_color'] for item in fpr_bar_data]
edgecolors = [item['color'] for item in fpr_bar_data]

bars = ax_bar.bar(
    names, fpr_vals, 
    color=colors, 
    edgecolor=edgecolors, 
    linewidth=2.0, 
    alpha=0.9, 
    width=0.55
)

for bar, item in zip(bars, fpr_bar_data):
    yval = bar.get_height()
    fontweight = 'bold' if item['is_target'] else 'normal'
    textcolor = item['color'] if item['is_target'] else '#4f5b66'
    
    ax_bar.text(
        bar.get_x() + bar.get_width() / 2.0, 
        yval + (max(fpr_vals) * 0.015), 
        f"{yval:.2f}", 
        ha='center', va='bottom', 
        fontsize=16, 
        fontweight=fontweight, 
        color=textcolor
    )

ax_bar.set_ylabel('False Positive Rate (%)', fontsize=20)
ax_bar.set_title('False Positive Rate (FPR) Comparison at Cutoff', fontsize=22, fontweight='bold')
ax_bar.tick_params(axis='x', labelsize=16, rotation=15)
ax_bar.tick_params(axis='y', labelsize=18)
ax_bar.set_ylim(0, max(fpr_vals) * 1.15)

fig_bar.tight_layout()
fig_bar.savefig(FPR_BAR_PLOT_FILE)
plt.close(fig_bar)
log_and_print(f"[PLOT] Saved FPR bar plot to {FPR_BAR_PLOT_FILE}")

# ==============================================================================
# 5) Significance Score 기반 Input Feature Distribution 플롯
# ==============================================================================
log_and_print("Generating Significance Score-based Input Features Distribution plot...")
df_raw['idf_per_L'] = df_raw['idf'] / df_raw['L']

score_col = None
for candidate in ['score_significance', 'score_Significance', 'score_NN_z', 'score_nn']:
    if candidate in df_raw.columns:
        score_col = candidate
        break
if score_col is None:
    score_col = 'score_NN_z'

sig_scores = df_raw[score_col].values
sig_thresh = globals().get('SIGNIFICANCE_THRESHOLD', globals().get('NN_THRESHOLD_LOGIT', 0.5))
log_and_print(f"[feature-dist] using column '{score_col}' with threshold {sig_thresh:.6f}")

sig_tp_mask = (df_raw['y'] == 1) & (sig_scores >= sig_thresh)
sig_fp_mask = (df_raw['y'] == 0) & (sig_scores >= sig_thresh)
sig_fn_mask = (df_raw['y'] == 1) & (sig_scores < sig_thresh)

feature_cols = ['idf', 'idf_per_L', 'tm_score', 'rmsd']
display_names = ['IDF Score', 'IDF / L (Score per Residue)', 'TM-Score', 'RMSD']

fig, axes = plt.subplots(1, 4, figsize=(24, 6)) 
axes = axes.flatten()

for i in range(len(feature_cols)):
    ax = axes[i]
    col = feature_cols[i]
    
    feat_tp = df_raw.loc[sig_tp_mask, col].dropna()
    feat_fp = df_raw.loc[sig_fp_mask, col].dropna()
    feat_fn = df_raw.loc[sig_fn_mask, col].dropna()
    
    min_val, max_val = np.nanmin(df_raw[col]), np.nanmax(df_raw[col])
    plot_max = 500 if col == 'idf' else max_val
    bins = np.linspace(min_val, plot_max, 50)

    ax.hist(feat_tp, bins=bins, alpha=0.6, color='#008080', label='TP (Correct Homol)')
    ax.hist(feat_fp, bins=bins, alpha=0.8, color='#e74c3c', label='FP (The "Tricked" cases)')
    ax.hist(feat_fn, bins=bins, alpha=0.4, color='#f39c12', label='FN (Missed Cases)')

    ax.set_xlim(min_val, plot_max)
    ax.set_title(f'{display_names[i]}\n(Significance Score Ref)', fontsize=18, fontweight='bold')
    ax.set_xlabel(f'{display_names[i]}', fontsize=18)
    ax.set_ylabel('Number of Samples', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(loc='upper right', fontsize=18)

fig.tight_layout()
fig.savefig(FEATURE_DIST_FILE)
plt.close(fig)

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    f.writelines(summary_lines)

log_and_print(f"Evaluation complete. Summary report saved to {RESULTS_FILE}")