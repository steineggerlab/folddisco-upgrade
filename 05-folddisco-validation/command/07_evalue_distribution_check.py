# Python script to analyze the distribution of E-values for self, similar, and non-similar hits across different query lengths (L) and plot the results.

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.stats import gaussian_kde
from scipy.stats import wasserstein_distance
from scipy.stats import median_abs_deviation

METRICS = ["IDF_score","RMSD","TM_score","GDT_TS","GDT_HA","Chamfer_distance","Hausdorff_distance"]

# ---------------------------------------------------------
# Receive METRIC_NUM from command line
# ---------------------------------------------------------
if len(sys.argv) >= 2:
    try:
        METRIC_NUM = int(sys.argv[1])
        if not (0 <= METRIC_NUM < len(METRICS)):
            raise ValueError
    except:
        print(f"[ERROR] Invalid METRIC_NUM argument: {sys.argv[1]}")
        sys.exit(1)
else:
    METRIC_NUM = 0

print(f"[INFO] Using METRIC_NUM={METRIC_NUM} ({METRICS[METRIC_NUM]})")

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')
plt.rc('font', size=15)

# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------
RESULT_DIR = "result"
ANA_DIR   = os.path.join(RESULT_DIR, "folddisco_results_evalues")
PNG_DIR    = os.path.join(RESULT_DIR, "analyses_plot_distribution")

os.makedirs(PNG_DIR, exist_ok=True)

EVAL_CUTOFF = 1
FILE_INFO = os.path.join(
    ANA_DIR,
    f"evalues_stats_total{METRIC_NUM}_{METRICS[METRIC_NUM]}.txt"
)

def smart_split(line):
    return line.rstrip("\n").split("\t")

# ---------------------------------------------------------
# Load data grouped by L
# ---------------------------------------------------------
def load_evalues_grouped_by_length():
    groups = {}   # L → { "self":[], "similar":[], "non":[] }

    if not os.path.exists(FILE_INFO):
        print(f"[ERROR] File not found: {FILE_INFO}")
        sys.exit(1)

    with open(FILE_INFO, "r") as f:
        header = f.readline()
        for line in f:
            cols = smart_split(line)
            if len(cols) < 8:
                continue

            try:
                L = int(cols[3])
                e_value = float(cols[5])
                is_self = int(cols[6])
                is_similar = int(cols[7])
            except ValueError:
                continue

            if L not in groups:
                groups[L] = {"self":[], "similar":[], "non":[]}

            if is_self == 1:
                groups[L]["self"].append(e_value)
            elif is_similar == 1:
                groups[L]["similar"].append(e_value)
            else:
                groups[L]["non"].append(e_value)

    return groups

def valid_for_kde(x):
    x = np.array(x, float)
    x = x[np.isfinite(x)]   # remove inf, nan
    return x if len(x) >= 2 else None


# ---------------------------------------------------------
# Plot per L
# ---------------------------------------------------------
def plot_single_length(L, arr_self, arr_similar, arr_non):

    arr_self    = np.array(arr_self, float)
    arr_self    = arr_self[np.isfinite(arr_self)]
    arr_similar = np.array(arr_similar, float)
    arr_similar = arr_similar[np.isfinite(arr_similar)]
    arr_non     = np.array(arr_non, float)
    arr_non     = arr_non[np.isfinite(arr_non)]

    if len(arr_self) == 0 or len(arr_similar) == 0 or len(arr_non) == 0:
        print(f"[WARN] L={L} array empty → skipping plotting for this L")
        return

    eps = 1e-300

    # --- e-value → -log10(e-value) ---
    x_self = -np.log10(arr_self + eps)
    x_sim  = -np.log10(arr_similar + eps)
    x_non  = -np.log10(arr_non + eps)

    # =========================================================
    # [수정됨] X축 범위 설정: Min은 -2로 고정, Max는 데이터에 따름
    # =========================================================
    all_data = np.concatenate([x_self, x_sim, x_non])
    
    if len(all_data) > 0:
        data_max = 50.0 if np.max(all_data) > 50.0 else np.max(all_data)
    else:
        data_max = 10.0

    # 오른쪽(최댓값)에만 여백 추가
    padding_right = (data_max - (-2)) * 0.05 
    if padding_right <= 0: padding_right = 1.0

    # [요청사항 반영] plot_min을 -2로 고정
    plot_min = 0.0
    plot_max = data_max + padding_right
    
    # 만약 데이터가 모두 -2보다 작아서 plot_max가 이상해질 경우 방지
    if plot_max <= plot_min:
        plot_max = plot_min + 5.0

    # KDE 계산용 좌표 (설정된 min, max 범위 내)
    xs = np.linspace(plot_min, plot_max, 800)

    # Wasserstein Distance 계산
    wd_sn  = wasserstein_distance(x_self, x_non)
    wd_ssi = wasserstein_distance(x_self, x_sim)
    wd_sin = wasserstein_distance(x_sim, x_non)

    # =========================================================
    # Robust Effect Size
    # =========================================================
    mad_sim = median_abs_deviation(x_sim, scale='normal')
    mad_non = median_abs_deviation(x_non, scale='normal')

    if mad_sim == 0: mad_sim = 1e-9
    if mad_non == 0: mad_non = 1e-9

    n1, n2 = len(x_sim), len(x_non)
    pooled_mad_sq = ((n1 - 1) * (mad_sim**2) + (n2 - 1) * (mad_non**2)) / (n1 + n2 - 2)
    pooled_robust_std = np.sqrt(pooled_mad_sq)

    if pooled_robust_std > 0:
        robust_effect_size = wd_sin / pooled_robust_std
    else:
        robust_effect_size = 0.0

    # --- Plot Setup ---
    plt.figure(figsize=(7,5))

    x_self_clean    = valid_for_kde(x_self)
    x_similar_clean = valid_for_kde(x_sim)
    x_non_clean     = valid_for_kde(x_non)

    # self KDE
    if x_self_clean is None:
        print(f"[WARN] L={L} self KDE skipped")
    else:
        kde_self = gaussian_kde(x_self_clean)
        ys_self = kde_self(xs)
        plt.plot(xs, ys_self, lw=2, label="self-hits")

    # non-self KDE
    if x_non_clean is None:
        print(f"[WARN] L={L} non-similar KDE skipped")
    else:
        kde_non = gaussian_kde(x_non_clean)
        ys_non = kde_non(xs)
        plt.plot(xs, ys_non, lw=2, label="non-similar-hits")

    # similar KDE
    if x_similar_clean is None:
        print(f"[WARN] L={L} similar KDE skipped")
    else:
        kde_sim = gaussian_kde(x_similar_clean)
        ys_sim = kde_sim(xs)
        plt.plot(xs, ys_sim, lw=2, label="similar-hits")

    # [수정] X축 범위를 -2부터 계산된 max까지 설정
    plt.xlim(plot_min, plot_max)
    plt.yscale("linear")   

    plt.xlabel("-log(e-value)", fontsize=20)
    plt.ylabel("Density", fontsize=20)
    plt.title(
        f"L={L}\n"
        f"sim vs non: WD={wd_sin:.2f}\n Robust Effect Size={robust_effect_size:.3f}"
    )

    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(PNG_DIR, f"evalue_dist_L{L}_{METRICS[METRIC_NUM]}.png")
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"[PLOT] Saved → {out_png}")

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    groups = load_evalues_grouped_by_length()

    print(f"[INFO] Found {len(groups)} length groups.")

    for L, g in sorted(groups.items()):
        if len(g["self"]) == 0 or len(g["similar"]) == 0 or len(g["non"]) == 0:
            print(f"[WARN] L={L} insufficient data → skipped")
            continue

        print(f"[INFO] Processing L={L}")
        plot_single_length(L, g["self"], g["similar"], g["non"])

    print("[DONE] All L plotted.")

if __name__ == "__main__":
    main()