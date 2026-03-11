# Python script to convert FoldDisco results to E-values using fitted parameters
import numpy as np
import os
import re
import sys

RESULT_DIR = "result/folddisco_results_analyses"
INDEX_DIR = "data/index_pdbs"
REF_DB_SIZE = 10546.0

# ---------------------------------------------------------
# DB Size 및 Metric 설정
# ---------------------------------------------------------
if os.path.exists(INDEX_DIR):
    N_INDEX = sum(1 for f in os.listdir(INDEX_DIR) if f.endswith(".pdb"))
    M_EFF = float(max(N_INDEX - 1, 1))
else:
    M_EFF = 4000.0

METRICS = ["IDF_score","RMSD","TM_score","GDT_TS","GDT_HA","Chamfer_distance","Hausdorff_distance"]
METRIC_NUM = int(sys.argv[1]) if len(sys.argv) >= 2 else 0

print(f"[INFO] Applying E-values using {METRICS[METRIC_NUM]}")
print(f"[INFO] Real DB Size (m): {M_EFF}, Reference DB Size: {REF_DB_SIZE}")

EVALUE_FILE = f"total_evalues_fitted_metric{METRIC_NUM}_{METRICS[METRIC_NUM]}.txt"

# ---------------------------------------------------------
# Mu/Lambda 테이블 로드
# ---------------------------------------------------------
def load_fitting_table():
    tbl = {}
    for L in range(2, 33):
        l_d = float(L)
        mu = 4.2161 * np.exp(l_d * 0.0489) + 3.6661
        lam = 0.28 * np.exp(l_d * -0.078) + 0.035
        tbl[L] = (mu, lam)
    return tbl

def parse_length_from_filename(path):
    m = re.search(r"length_(\d+)_combined", os.path.basename(path))
    return int(m.group(1)) if m else None

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    MU_TABLE = load_fitting_table()
    files = [
        os.path.join(RESULT_DIR, f)
        for f in os.listdir(RESULT_DIR)
        if f.startswith("length_") and "combined" in f
    ]

    print(f"{len(files)} files to process.")
    out_path = os.path.join(RESULT_DIR, EVALUE_FILE)

    # [최적화] 출력 파일을 루프 밖에서 한 번만 열어 유지
    with open(out_path, "w", encoding='utf-8') as out:
        out.write(f"query_num\tsource\tid\tlen\t{METRICS[METRIC_NUM]}\te_value\n")

        for file_path in files:
            L = parse_length_from_filename(file_path)
            if L is None or L not in MU_TABLE:
                continue

            mu_L, lam_L = MU_TABLE[L]
            l_d = float(L)
            
            # [최적화] 고정 계산값은 내부 루프 밖으로 이동
            k_val = np.exp(lam_L * mu_L) / REF_DB_SIZE
            k_m_l = k_val * M_EFF * l_d

            with open(file_path, "r", encoding='utf-8') as f:
                header = f.readline()
                for line in f:
                    cols = line.rstrip("\n").split("\t")
                    if len(cols) < 6:
                        continue

                    try:
                        if METRIC_NUM == 0:
                            score = float(cols[3])
                        elif METRIC_NUM == 1:
                            score = -np.log(float(cols[4]) + 1)
                        else:
                            score = float(cols[3])
                        
                        # E-value 계산
                        e_raw = k_m_l * np.exp(-lam_L * score)
                        e = (e_raw * M_EFF) / (e_raw + M_EFF)

                        # [유지] file_path 전체 경로를 사용하여 후속 코드 호환성 확보
                        out.write(f"{file_path}\t{cols[0]}\t{cols[1]}\t{L}\t{score:.6f}\t{e:.3e}\n")
                        
                    except (ValueError, ZeroDivisionError, IndexError):
                        continue

            print(f"Processed: {os.path.basename(file_path)}")

    print(f"[DONE] Final E-value file written → {out_path}")

if __name__ == "__main__":
    main()