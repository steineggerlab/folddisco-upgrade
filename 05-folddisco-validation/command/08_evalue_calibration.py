# Python script to calibrate E-values by comparing observed distributions to theoretical expectations and adjusting parameters accordingly.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 파일 경로 및 디렉토리 설정
FILE_INFO = "result/folddisco_results_evalues/evalues_stats_total0_IDF_score.txt"
PNG_DIR = "result/analyses_plot_fitting"
RMSE_FILE = "result/analyses_plot_fitting/rmse_values_evalues.txt"

if not os.path.exists(PNG_DIR):
    os.makedirs(PNG_DIR, exist_ok=True)

def pp_plot_data_on_ax(ax, x, length):
    """
    ax(subplot) 위에 PP-plot을 그리는 함수
    """
    # --- [1] QQ-plot Data Preparation ---
    # 0 이하의 값은 로그 스케일이나 계산을 위해 아주 작은 양수로 대체
    min_pos = x[x > 0].min() if np.any(x > 0) else 1e-300
    safe_x = np.where(x <= 0, min_pos, x)
    
    obs_scores = np.sort(safe_x)
    n = len(obs_scores)
    
    if n == 0:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center')
        return 0 # RMSE 기본값 반환

    # 이론적 E-value 계산
    quantiles = np.arange(1, n + 1) / (n + 1)
    theo_values = -np.log(1 - quantiles)
    theo_values = np.sort(theo_values)

    # --- [2] RMSE Calculation (관심 영역: 0.001 ~ 1) ---
    target_mask = (obs_scores >= 0.001) & (obs_scores <= 1)
    if np.any(target_mask):
        mse = np.mean((obs_scores[target_mask] - theo_values[target_mask]) ** 2)
        rmse_roi = np.sqrt(mse)
    else:
        rmse_roi = 0

    # --- [3] Plotting ---
    ax.plot(theo_values, obs_scores, linewidth=1.5, label='Obs')
    ax.plot([1e-5, 1], [1e-5, 1], 'r--', linewidth=1.5, label='Theory')

    # Zoom-in 설정
    ax.set_xlim(0.0001, 1)
    ax.set_ylim(0.0001, 1)

    # Label 및 Title
    ax.set_xlabel("Theo (E-value)", fontsize=8)
    ax.set_ylabel("Obs (E-value)", fontsize=8)
    
    title_str = f"Len: {length} | RMSE={rmse_roi:.3f}" if length != "TOTAL" else f"TOTAL | RMSE={rmse_roi:.3f}"
    ax.set_title(title_str, fontsize=9, fontweight='bold')
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=7)
    
    return rmse_roi

def main():
    # 1. 데이터 로드 및 전처리
    try:
        df = pd.read_csv(FILE_INFO, sep="\t")
    except Exception:
        # 더미 데이터 (테스트용)
        np.random.seed(42)
        df = pd.DataFrame({
            "len": np.random.randint(2, 33, 2000),
            "e_value": np.random.exponential(scale=1.0, size=2000)
        })

    df["e_value"] = pd.to_numeric(df["e_value"], errors='coerce')
    df = df.dropna(subset=["e_value", "len"])
    df = df[df["len"] > 0]

    # [핵심 수정] 각 행의 길이에 2를 곱한 값을 param으로 사용하여 e_value를 나눔
    #df["e_value_scaled"] = df["e_value"] / 2
    df["e_value_scaled"] = df["e_value"] / (2 * df["len"])

    # 2. Canvas 설정
    rows, cols = 4, 8
    fig, axes = plt.subplots(rows, cols, figsize=(24, 16))
    axes_flat = axes.flatten()

    target_lengths = list(range(2, 33)) + ["TOTAL"]
    rmse_results = []

    for i, length in enumerate(target_lengths):
        if i >= len(axes_flat): break 
        ax = axes_flat[i]
        
        if length == "TOTAL":
            data_to_plot = df["e_value_scaled"].values
            current_p = "2*len"
            for spine in ax.spines.values():
                spine.set_edgecolor('blue')
                spine.set_linewidth(2)
        else:
            data_to_plot = df[df["len"] == length]["e_value_scaled"].values
            current_p = float(2 * length)
        
        rmse = pp_plot_data_on_ax(ax, data_to_plot, length)
        rmse_results.append((length, current_p, rmse))

    # 4. RMSE 결과 저장
    with open(RMSE_FILE, "w") as f_rmse:
        f_rmse.write("Length\tUsed_Param\tRMSE\n")
        for length, p_val, rmse in rmse_results:
            f_rmse.write(f"{length}\t{p_val}\t{rmse:.6f}\n")

    # 5. 이미지 저장
    plt.tight_layout()
    save_path = os.path.join(PNG_DIR, "combined_pp_plots_scaled.jpg")
    plt.savefig(save_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    main()