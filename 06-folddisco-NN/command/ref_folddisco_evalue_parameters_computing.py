# Python script to analyze Folddisco results and compute parameters for e-values

import numpy as np
import os
import re
import sys
import math
from multiprocessing import Pool, cpu_count

# ---------------------------------------------------------
# [설정] 전체 DB 크기 (보정에 사용)
# ---------------------------------------------------------
RESULT_DIR = "result_expanded4/folddisco_results_analyses"
INPUT_FILE = os.path.join(RESULT_DIR, "total_combined.txt") 
PDB_FILE   = "data/domain-list-index.txt"

METRICS = ["IDF_score","RMSD","TM_score","GDT_TS","GDT_HA","Chamfer_distance","Hausdorff_distance"]
SUBGROUP_DICT = {}

if len(sys.argv) >= 2:
    METRIC_NUM = int(sys.argv[1])
else:
    METRIC_NUM = 0

print(f"[INFO] Fitting μ/λ for metric = {METRICS[METRIC_NUM]}")
print(f"[INFO] Method: Fit lambda on raw hits, Shift mu by log(N_obs/N_total)")

FITTING_INFO = f"evalue_mu_lambda_per_length_list{METRIC_NUM}_{METRICS[METRIC_NUM]}.txt"
FITTING_FILE = os.path.join(RESULT_DIR, FITTING_INFO)

# ---------------------------------------------------------
# Utilities & Fitting Functions
# ---------------------------------------------------------
def smart_split(line):
    return line.rstrip("\n").split("\t")

def subgroup_detector(line):
    cols = smart_split(line)
    if len(cols) < 4: return
    pdb_id = cols[0]
    subgroup = cols[1] + cols[2] + cols[3]
    SUBGROUP_DICT[pdb_id] = subgroup

def _lawless416(x, lam):
    x_safe = np.array(x, dtype=np.float64)
    args = -lam * x_safe
    ex = np.exp(args)
    esum = ex.sum()
    if esum == 0: return 0, 0 
    xesum = (x_safe * ex).sum()
    xxesum = (x_safe * x_safe * ex).sum()
    f  = 1 / lam - x_safe.mean() + xesum / esum
    df = (xesum/esum)**2 - xxesum/esum - 1/(lam**2)
    return f, df

def _newton_with_bisect(f_df, lam0=0.2, tol=1e-5):
    lam = max(lam0, 1e-6)
    for _ in range(80):
        try:
            f, df = f_df(lam)
            if df == 0: break 
            if abs(f) < tol: return lam
            lam = max(lam - f/df, 1e-6)
        except: break
    L, R = 1e-6, 100
    for _ in range(150):
        M = (L + R) / 2
        fM, _ = f_df(M)
        if abs(fM) < tol: return M
        if fM > 0: L = M
        else:      R = M
    return (L + R) / 2

def evd_mle_full(scores):
    """
    순수 Raw 데이터에 대해 MLE를 수행하여 mu, lambda를 구합니다.
    """
    x = np.asarray(scores, float)
    x = x[np.isfinite(x)]
    if x.size < 5: return np.nan, np.nan
    if np.std(x) < 1e-4: return np.nan, np.nan
        
    lam = _newton_with_bisect(lambda L: _lawless416(x, L))
    if not np.isfinite(lam) or lam <= 0: return np.nan, np.nan
    m = np.mean(np.exp(-lam * x))
    if m <= 0: return np.nan, np.nan
    mu = -math.log(m) / lam
    return mu, lam

def evd_mle_tail_raw(scores, top_fraction=1.0):
    """
    관측된 데이터의 상위 N%만 사용하여 Tail Fitting (Raw Parameter 도출)
    """
    scores = np.sort(scores)
    cutoff_idx = int(len(scores) * (1 - top_fraction))
    tail_scores = scores[cutoff_idx:]

    if len(tail_scores) < 10:
        tail_scores = scores # 데이터 적으면 전체 사용

    mu, lam = evd_mle_full(tail_scores)

    return mu, lam, len(tail_scores)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("[INFO] Loading subgroup info...")
    try:
        with open(PDB_FILE, "r") as f:
            for line in f:
                subgroup_detector(line)
    except FileNotFoundError:
        pass

    scores_by_length = {}
    print(f"[INFO] Reading {INPUT_FILE}...")
    
    count = 0
    with open(INPUT_FILE, "r") as f:
        header = f.readline()
        for line in f:
            cols = smart_split(line)
            if len(cols) < 5: 
                continue
            
            # 1. Length Parsing
            try:
                if 'length_' not in cols[0] or 'length_0_' in cols[0]: continue
                L_str = cols[0].replace("length_", "").replace("_combined.txt", "")
                query_col = cols[-2]
                valid_queries = [x for x in query_col.split(',') if x != '_']
                L = len(valid_queries)
                if L == 0 or L == 1: continue
            except ValueError:
                continue
            
            # 2. Filtering
            q = cols[1].replace("output_", "").replace(".txt", "") 
            t = os.path.basename(cols[2]).replace(".pdb", "")
            if q == t:
                continue
            gq = SUBGROUP_DICT.get(q, "")
            gt = SUBGROUP_DICT.get(t, "")
            if gq != "" and gq == gt: 
                continue
            
            # 3. Score Parsing (IDF=4, RMSD=5)
            try:
                idf  = float(cols[4]) 
                rmsd = float(cols[5]) 
                
                if METRIC_NUM == 0: score = idf
                elif METRIC_NUM == 1: score = -np.log(rmsd + 1)
                else: score = idf 
                
                if L not in scores_by_length:
                    scores_by_length[L] = []
                scores_by_length[L].append(score)
            except ValueError:
                print(f"[WARN] Invalid score value: {cols[4]}, {cols[5]}")
                continue
            
            count += 1
            if count % 1000000 == 0:
                print(f"...processed {count} lines")

    print(f"[INFO] Total valid hits: {count}")
    
    with open(FITTING_FILE, "w") as out:
        out.write("Length\tmu_raw\tlam_raw\tmu_cut\tlam_cut\n")
        
        sorted_keys = sorted(scores_by_length.keys())
        for L in sorted_keys:
            if L > 32:
                continue
            scores = scores_by_length[L]
            n_obs = len(scores)
            
            # 1. Raw Data로 lambda, mu 계산 (순수 분포의 모양 파악)
            mu_raw, lam_raw, n_tail = evd_mle_tail_raw(scores, top_fraction=1.0)
            mu_half, lam_half, n_tail_half = evd_mle_tail_raw(scores, top_fraction=0.5)
            mu_cut, lam_cut, n_tail_cut = evd_mle_tail_raw(scores, top_fraction=0.1)                                
            if np.isfinite(mu_cut) and np.isfinite(lam_cut) and lam_cut > 0:
                out.write(f"{L}\t{mu_raw:.6f}\t{lam_raw:.6f}\t{mu_cut:.6f}\t{lam_cut:.6f}\n")
                print(f"[OK] L={L} mu_raw={mu_raw:.6f}, mu_half={mu_half:.6f}, mu_cut={mu_cut:.6f}, lam_raw = {lam_raw:.6f}, slam={lam_cut:.6f} (N_obs={n_obs})")
            else:
                print(f"[FAIL] L={L} fit failed.")

    print(f"[DONE] Saved to {FITTING_FILE}")

if __name__ == "__main__":
    main()