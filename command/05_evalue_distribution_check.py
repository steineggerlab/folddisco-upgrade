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
RESULT_DIR = "result_whole"
ANA_DIR   = os.path.join(RESULT_DIR, "folddisco_results_evalues")
PNG_DIR    = os.path.join(RESULT_DIR, "analyses_plot_test")

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

    with open(FILE_INFO, "r") as f:
        header = f.readline()
        for line in f:
            cols = smart_split(line)
            if len(cols) < 8:
                continue

            L = int(cols[3])
            e_value = float(cols[5])
            is_self = int(cols[6])
            is_similar = int(cols[7])

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
    arr_self = arr_self[np.isfinite(arr_self)]
    arr_similar = np.array(arr_similar, float)
    arr_similar = arr_similar[np.isfinite(arr_similar)]
    arr_non     = np.array(arr_non, float)
    arr_non = arr_non[np.isfinite(arr_non)]

    if len(arr_self) == 0 or len(arr_similar) == 0 or len(arr_non) == 0:
        print(f"[WARN] L={L} array empty → skipping plotting for this L")
        return

    arr_self_safe = np.where(arr_self <= 0, 1e-300, arr_self)
    arr_similar_safe = np.where(arr_similar <= 0, 1e-300, arr_similar)
    arr_non_safe = np.where(arr_non <= 0, 1e-300, arr_non)

    min_self_e = np.min(arr_self_safe)
    min_similar_e = np.min(arr_similar_safe)
    min_non_e = np.min(arr_non_safe)

    min_self_e = min(min_self_e, min_similar_e, min_non_e)

    if min_self_e <= 0:
        min_self_e = 1e-300

    # -log10 변환
    max_x = -np.log10(min_self_e)
    if max_x < 1:
        max_x = 1.0  # 최소한 1 이상의 x축 확보

    if not np.isfinite(max_x):
        max_x = 10.0   # 기본 범위


    # 0이나 너무 작은 값 대비용 eps
    eps = 1e-300

    # --- e-value → -log10(e-value) 로 변환 ---
    x_self = -np.log10(arr_self + eps)
    x_sim  = -np.log10(arr_similar + eps)
    x_non  = -np.log10(arr_non + eps)

    bins = np.linspace(0, max_x, 200)

    hs, edges   = np.histogram(x_self,    bins=bins, density=True)
    hsim, _     = np.histogram(x_sim,     bins=bins, density=True)
    hn, _       = np.histogram(x_non,     bins=bins, density=True)
    centers     = (edges[:-1] + edges[1:]) / 2

    wd_sn  = wasserstein_distance(x_self, x_non)
    wd_ssi = wasserstein_distance(x_self, x_sim)
    wd_sin = wasserstein_distance(x_sim, x_non)

# =========================================================
    # [수정됨] Robust Effect Size (Non-normal Distribution용)
    # =========================================================
    # 1. MAD 계산 (scale='normal' 옵션을 쓰면 1.4826을 자동으로 곱해줌)
    #    scipy 1.5.0 이상에서는 scale='normal' 사용 가능, 구버전이면 수동으로 * 1.4826
    mad_sim = median_abs_deviation(x_sim, scale='normal')
    mad_non = median_abs_deviation(x_non, scale='normal')

    # 만약 MAD가 0인 경우(데이터가 모두 같은 값), 최소한의 값을 주어 에러 방지
    if mad_sim == 0: mad_sim = 1e-9
    if mad_non == 0: mad_non = 1e-9

    # 2. Pooled Robust Deviation (분산 합동 공식과 유사하게 접근)
    n1, n2 = len(x_sim), len(x_non)
    # MAD를 제곱해서 분산처럼 취급하여 합친 후 다시 제곱근
    pooled_mad_sq = ((n1 - 1) * (mad_sim**2) + (n2 - 1) * (mad_non**2)) / (n1 + n2 - 2)
    pooled_robust_std = np.sqrt(pooled_mad_sq)

    # 3. Effect Size 계산
    if pooled_robust_std > 0:
        robust_effect_size = wd_sin / pooled_robust_std
    else:
        robust_effect_size = 0.0

    # 해석 (Robust 버전에서도 기준은 비슷하게 가져갑니다)
    if robust_effect_size > 0.5: effect_text = "Large"
    elif robust_effect_size > 0.3: effect_text = "Medium"
    elif robust_effect_size > 0.1: effect_text = "Small"
    else: effect_text = "Negligible"
    # =========================================================

    # --- Plot histogram ---
    ymin = min(hs.min(), hsim.min(), hn.min())
    ymax = max(hs.max(), hsim.max(), hn.max())

    plt.figure(figsize=(7,5))
    width = np.diff(edges)
    offset = width * 0.25


    #plt.bar(centers - offset, hs, width=width*0.25, alpha=0.8, label="self")
    #plt.bar(centers, hsim, width=width*0.25, alpha=0.8, label="similar")
    #plt.bar(centers + offset, hn, width=width*0.25, alpha=0.8, label="non-self")

    x_self_clean    = valid_for_kde(x_self)
    x_similar_clean = valid_for_kde(x_sim)
    x_non_clean     = valid_for_kde(x_non)

    xs = np.linspace(0, max_x, 800)

# self KDE
    if x_self_clean is None:
        print(f"[WARN] L={L} self KDE skipped (nan/inf or <2 samples)")
    else:
        kde_self = gaussian_kde(x_self_clean)
        ys_self = kde_self(xs)
        plt.plot(xs, ys_self, lw=2, label="self-hits")

    # non-self KDE
    if x_non_clean is None:
        print(f"[WARN] L={L} non-similar KDE skipped (nan/inf or <2 samples)")
    else:
        kde_non = gaussian_kde(x_non_clean)
        ys_non = kde_non(xs)
        plt.plot(xs, ys_non, lw=2, label="non-similar-hits")

        # similar KDE
    if x_similar_clean is None:
        print(f"[WARN] L={L} similar KDE skipped (nan/inf or <2 samples)")
    else:
        kde_sim = gaussian_kde(x_similar_clean)
        ys_sim = kde_sim(xs)
        plt.plot(xs, ys_sim, lw=2, label="similar-hits")


    plt.xlim(0, max_x)
    plt.yscale("linear")   

    plt.xlabel("-log(e-value)", fontsize=20)
    plt.ylabel("Density", fontsize=20)
    plt.title(
        f"L={L}\n"
        f"similar vs non: WD={wd_sin:.2f}\n Robust Effect Size={robust_effect_size:.3f}"
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
