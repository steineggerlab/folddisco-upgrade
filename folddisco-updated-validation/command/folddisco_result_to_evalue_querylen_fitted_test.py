# Python script to analyze Folddisco results and make score distribution

import numpy as np
import matplotlib.pyplot as plt
import os
import re
import sys
import math
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from scipy.stats import ks_2samp

RESULT_DIR = "result_whole/folddisco_results_analyses"
PNG_DIR    = "result_whole/analyses_plot"
FILE_INFO  = "total_combined.txt"
PDB_FILE = "data/domain-list-index.txt"

INDEX_DIR = "data/index_pdbs"
N_INDEX = sum(1 for f in os.listdir(INDEX_DIR) if f.endswith(".pdb"))
M_EFF = max(N_INDEX - 1, 1)

METRICS = ["IDF_score","RMSD","TM_score","GDT_TS","GDT_HA","Chamfer_distance","Hausdorff_distance"]

SUBGROUP_DICT = {}

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
        print(f"Usage: python3 script.py <metric_num>")
        print(f"metric_num must be 0–{len(METRICS)-1}")
        sys.exit(1)
else:
    METRIC_NUM = 0   # default = IDF-score

print(f"[INFO] Using METRIC_NUM={METRIC_NUM} ({METRICS[METRIC_NUM]})")

EVALUE_FILE = f"test_total_evalues_fitted_metric{METRIC_NUM}_{METRICS[METRIC_NUM]}.txt"
STAT_FILE = f"test_pvalues_stat_{METRICS[METRIC_NUM]}.txt"


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------

def smart_split(line):
    return line.rstrip("\n").split("\t")


def _lawless416(x, lam):
    ex = np.exp(-lam * x)
    esum = ex.sum()
    xesum = (x * ex).sum()
    xxesum = (x * x * ex).sum()
    f  = 1 / lam - x.mean() + xesum / esum
    df = (xesum/esum)**2 - xxesum/esum - 1/(lam**2)
    return f, df


def _newton_with_bisect(f_df, lam0=0.2, tol=1e-5):
    lam = max(lam0, 1e-6)
    for _ in range(80):
        f, df = f_df(lam)
        if abs(f) < tol:
            return lam
        lam = max(lam - f/df, 1e-6)

    L, R = 1e-6, 100
    for _ in range(150):
        M = (L + R) / 2
        fM, _ = f_df(M)
        if abs(fM) < tol:
            return M
        if fM > 0:
            L = M
        else:
            R = M
    return (L + R) / 2


def evd_mle_full(scores):
    x = np.asarray(scores, float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        print("small x size")
        return np.nan, np.nan

    std = np.std(x)
    uniq = np.unique(x)

    if np.allclose(x, x[0]) or std < 1e-4:
        return np.nan, np.nan


    lam = _newton_with_bisect(lambda L: _lawless416(x, L))
    if not np.isfinite(lam) or lam <= 0:
        print("lambda weird")
        return np.nan, np.nan

    m = np.mean(np.exp(-lam * x))
    if m <= 0:
        print("mean minus")
        return np.nan, np.nan

    mu = -math.log(m) / lam
    if not np.isfinite(mu):
        print("mu infinite")
        return np.nan, np.nan

    return mu, lam


def evd_sf(x, mu, lam):
    x = np.asarray(x, float)
    y = lam * (x - mu)
    t = np.exp(-y)
    sf = 1 - np.exp(-t)
    sf[y > 20] = np.exp(-y[y > 20])
    return sf


def parse_length_from_filename(path):
    fname = os.path.basename(path)
    m = re.search(r"length_(\d+)_combined", fname)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse length from filename: {fname}")

# ---------------------------------------------------------
# Distinguishing self / similar / non-similar hits
# ---------------------------------------------------------
def subgroup_detector(line):
    cols = smart_split(line)
    if len(cols) < 2:
        return None

    pdb_id = cols[0]
    subgroup = cols[1] + cols[2] + cols[3]

    SUBGROUP_DICT[pdb_id] = subgroup
    return None

# ---------------------------------------------------------
# Worker: compute p-values + plot → return only rows
# ---------------------------------------------------------

def per_row_output(file_path):

    L_fixed = parse_length_from_filename(file_path)
    print(f"[INFO] Processing {file_path} (L={L_fixed})")

    rows = []
    scores_L = []   # <<==== 길이 단일 score bucket

    # -------------------------------
    # STEP 1: Load rows
    # -------------------------------
    with open(file_path) as f:
        header = f.readline()
        for line in f:
            cols = smart_split(line)
            if len(cols) < 14:
                continue

            key = cols[0]
            target = cols[1]

            # extract metrics
            idf = float(cols[3])
            rmsd = float(cols[4])
            tm = float(cols[7])
            gdt_ts = float(cols[8])
            gdt_ha = float(cols[9])
            chamf = float(cols[10])
            hausd = float(cols[11])

            # score selection
            if METRIC_NUM == 0:
                score = idf
            elif METRIC_NUM == 1:
                score = -np.log(rmsd + 1)
            elif METRIC_NUM == 2:
                score = -np.log(1 - tm + 1e-6)
            elif METRIC_NUM == 3:
                score = -np.log(1 - gdt_ts + 1e-6)
            elif METRIC_NUM == 4:
                score = -np.log(1 - gdt_ha + 1e-6)
            elif METRIC_NUM == 5:
                score = np.log(chamf + 1)
            elif METRIC_NUM == 6:
                score = np.log(hausd + 1)

            scores_L.append(score)
            rows.append(cols)

    print(f"[INFO] {file_path} - STEP 1 done (loaded {len(rows)} rows, total scores={len(scores_L)})")

    # -------------------------------
    # STEP 2: Fit μ(L), λ(L) directly from all scores of this length
    # -------------------------------
    if len(scores_L) >= 5:
        mu_L, lam_L = evd_mle_full(scores_L)
    else:
        mu_L, lam_L = np.nan, np.nan

    if (not np.isfinite(mu_L)) or (not np.isfinite(lam_L)) or lam_L <= 0:
        print(f"[WARN] EVD fitting failed for L={L_fixed}, using fallback values.")
        mu_L, lam_L = np.nan, np.nan

    print(f"[INFO] {file_path} - STEP 2 done (μ={mu_L}, λ={lam_L})")

    # -------------------------------
    # STEP 3: Compute p-values for each row
    # -------------------------------
    out_rows = []
    p_self = []
    p_similar = []
    p_non  = []

    for cols in rows:
        key = cols[0]
        target = cols[1]

        try:
            idf = float(cols[3])
            rmsd = float(cols[4])
            tm = float(cols[7])
            gdt_ts = float(cols[8])
            gdt_ha = float(cols[9])
            chamf = float(cols[10])
            hausd = float(cols[11])

            if METRIC_NUM == 0:
                score = idf
            elif METRIC_NUM == 1:
                score = -np.log(rmsd + 1)
            elif METRIC_NUM == 2:
                score = -np.log(1 - tm + 1e-6)
            elif METRIC_NUM == 3:
                score = -np.log(1 - gdt_ts + 1e-6)
            elif METRIC_NUM == 4:
                score = -np.log(1 - gdt_ha + 1e-6)
            elif METRIC_NUM == 5:
                score = np.log(chamf + 1)
            elif METRIC_NUM == 6:
                score = np.log(hausd + 1)
        except:
            continue

        # --- p-value using μ(L), λ(L)
        if np.isfinite(mu_L) and np.isfinite(lam_L) and lam_L > 0:
            p = evd_sf([score], mu_L, lam_L)[0]
            p = float(np.clip(p, 1e-300, 1.0))
        else:
            p = 1.0

        # e-value: M = number of total scores for this length (not key)
        M = len(scores_L)
        e = M * p

        out_rows.append(
            f"{file_path}\t{key}\t{target}\t{L_fixed}\t{score:.6f}\t{e:.3e}\n"
        )

        # ---------------------------
        # classify: self / similar / non
        # ---------------------------
        q = key.replace("output_","").replace(".txt","")
        t = os.path.basename(target).replace(".pdb","")

        if q == t:
            p_self.append(p)
        elif SUBGROUP_DICT.get(q, "") == SUBGROUP_DICT.get(t, "") and SUBGROUP_DICT.get(q, "") != "":
            p_similar.append(p)
        else:
            p_non.append(p)

    print(f"[INFO] {file_path} - STEP 3 done (self={len(p_self)}, similar={len(p_similar)}, non={len(p_non)})")

    # -------------------------------
    # (No STEP 4 plot here, skip)
    # -------------------------------
    return out_rows


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    with open(PDB_FILE, 'r') as data:
        for data_line in data.readlines():
            subgroup_detector(data_line)

    files = [
        os.path.join(RESULT_DIR, f)
        for f in os.listdir(RESULT_DIR)
        if "combined" in f and f.startswith("length_")
    ]

    print(f"[INFO] Found {len(files)} length groups.")

    nproc = min(cpu_count(), len(files))
    print(f"[INFO] Using {nproc} processes")

    with Pool(nproc) as pool:
        results = pool.map(per_row_output, files)

    # flatten
    all_rows = [line for sub in results for line in sub]

    # Write SINGLE output file
    out_path = os.path.join(RESULT_DIR, EVALUE_FILE)
    with open(out_path, "w") as out:
        out.write(f"query_num\tsource\tid\tlen\t{METRICS[METRIC_NUM]}\te_value\n")
        for line in all_rows:
            out.write(line)

    print(f"[DONE] Saved unified EVALUE file → {out_path}")


if __name__ == "__main__":
    main()
