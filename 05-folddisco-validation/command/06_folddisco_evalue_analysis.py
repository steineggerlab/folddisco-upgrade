# Python script to analyze Folddisco results and compute parameters for e-values

import os
import re
from collections import defaultdict
import sys
import numpy as np
import matplotlib.pyplot as plt


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
        print(f"[ERROR] Invalid METRIC_NUM arg: {sys.argv[1]}")
        sys.exit(1)
else:
    METRIC_NUM = 0

print(f"[INFO] Using METRIC_NUM={METRIC_NUM} ({METRICS[METRIC_NUM]})")

EVAL_CUTOFF = 1

EVALUE_FILE = f"total_evalues_fitted_metric{METRIC_NUM}_{METRICS[METRIC_NUM]}.txt"
TOTAL_FILE = f"evalues_stats_total{METRIC_NUM}_{METRICS[METRIC_NUM]}.txt"
STAT_FILE = f"evalues_stats_{METRICS[METRIC_NUM]}_{EVAL_CUTOFF}.txt"
LOG = f"summary_{METRICS[METRIC_NUM]}_{EVAL_CUTOFF}.txt"

DATA_DIR = "result_expanded/folddisco_results_analyses"
RESULT_DIR = "result_expanded/folddisco_results_evalues"
DATA_FILE = os.path.join(DATA_DIR, EVALUE_FILE)
PDB_FILE = "data/domain-list-index.txt"
RESULT_FILE = os.path.join(RESULT_DIR, STAT_FILE)
RESULT_TOTAL_FILE = os.path.join(RESULT_DIR, TOTAL_FILE)
LOG_FILE = os.path.join(RESULT_DIR, LOG)
os.makedirs(RESULT_DIR, exist_ok = True)

def log_and_print(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

# ---------------------------------------------------------
# Load subgroup dictionary
# ---------------------------------------------------------
def subgroup_detector(line):
    cols = line.strip().split("\t")
    if len(cols) < 4:
        return
    pdb_id = cols[0]
    subgroup = cols[1] + cols[2] + cols[3]
    SUBGROUP_DICT[pdb_id] = subgroup

with open(PDB_FILE) as f:
    for line in f:
        subgroup_detector(line)


# ---------------------------------------------------------
# e-value fixer
# ---------------------------------------------------------
def fix_evalue(val):
    m = re.match(r"([0-9]*\.[0-9]+)\.[0-9]+(e[\+\-]?[0-9]+)", val)
    if m:
        return m.group(1) + m.group(2)
    return val

# ---------------------------------------------------------
# Storage
# ---------------------------------------------------------
matches_by_qlen = defaultdict(list)     # match-level
queries_by_qlen = defaultdict(set)      # query-level
sig_queries_by_qlen = defaultdict(set)  # query-level positives
self_queries_by_qlen = defaultdict(set)
sim_queries_by_qlen = defaultdict(set)


# ---------------------------------------------------------
# Parsing phase
# ---------------------------------------------------------
with open(DATA_FILE) as f:
    lines = f.read().splitlines()

print("Opened data file successfully")

header = lines[0]
filtered_rows = []
all_rows = []


for line in lines[1:]:
    cols = line.split()
    if len(cols) < 6:
        continue

    qnum, query, index, node_count, tm_score, e_raw = cols[:6]

    m = re.search(r"/length_(\d+)_combined", qnum)
    if not m:
        continue
    qlen = m.group(1)

    queries_by_qlen[qlen].add(query)

    e_val = fix_evalue(e_raw)
    try:
        e_val_f = float(e_val)
    except:
        continue

    q_short = query.replace("output_", "").replace(".txt", "")
    idx_short = index.replace("data/index_pdbs/", "").replace(".pdb", "")

    is_self = (q_short == idx_short)
    is_similar = (
        SUBGROUP_DICT.get(q_short, "") != "" and
        SUBGROUP_DICT.get(q_short) == SUBGROUP_DICT.get(idx_short) and
        q_short != idx_short
    )
    is_sig = (e_val_f < EVAL_CUTOFF)

    # save match
    matches_by_qlen[qlen].append({
        "query": q_short,
        "index": idx_short,
        "evalue": e_val_f,
        "is_self": is_self,
        "is_similar": is_similar,
        "is_sig": is_sig,
        "raw": line 
    })

    if is_sig:
        filtered_rows.append(line)
        all_rows.append(line)
        sig_queries_by_qlen[qlen].add(query)

        if is_self:
            self_queries_by_qlen[qlen].add(query)
        elif is_similar:
            sim_queries_by_qlen[qlen].add(query)
    else:
        all_rows.append(line)

print("Parsing done.")

# ---------------------------------------------------------
# Save significant match rows
# ---------------------------------------------------------
with open(RESULT_FILE, "w") as out:
    out.write(header + "\n")
    for l in filtered_rows:
        out.write(l + "\n")

with open(RESULT_TOTAL_FILE, "w") as out:
    out.write(header + "\tis_self\tis_similar\n")

    for qlen in sorted(matches_by_qlen.keys(), key=lambda x: int(x)):
        for m in matches_by_qlen[qlen]:
            out.write(
                f"{m['raw']}\t{int(m['is_self'])}\t{int(m['is_similar'])}\n"
            )

log_and_print(f"[OK] Saved {len(filtered_rows)} significant match rows")


# ---------------------------------------------------------
# Summary per qlen
# ---------------------------------------------------------
def match_counts(match_list):
    total_self = sum(1 for m in match_list if m["is_self"])
    total_sim  = sum(1 for m in match_list if m["is_similar"])
    total_any  = sum(1 for m in match_list if m["is_self"] or m["is_similar"])
    total_sig = sum(1 for m in match_list if m["is_sig"])

    sig_self = sum(1 for m in match_list if m["is_sig"] and m["is_self"])
    sig_sim  = sum(1 for m in match_list if m["is_sig"] and m["is_similar"])
    sig_any  = sum(1 for m in match_list if m["is_sig"] and (m["is_self"] or m["is_similar"]))
    return total_self, total_sim, total_any, total_sig, sig_self, sig_sim, sig_any


# ---------------------------------------------------------
# Combined query-level precision + match-level recall
# ---------------------------------------------------------
def compute_confusion_matrix(match_list, ONLY_SELF=True):
    """
    match-level confusion matrix
    ground truth: self OR similar = positive
    prediction: significant = positive
    """

    if ONLY_SELF:
        TP = sum(1 for m in match_list if m["is_sig"] and m["is_self"])
        FP = sum(1 for m in match_list if m["is_sig"] and not m["is_self"])
        FN = sum(1 for m in match_list if m["is_self"] and not m["is_sig"])
        TN = sum(1 for m in match_list if (not m["is_self"]) and not m["is_sig"])

    else:
        TP = sum(1 for m in match_list if m["is_sig"] and (m["is_self"] or m["is_similar"]))
        FP = sum(1 for m in match_list if m["is_sig"] and not (m["is_self"] or m["is_similar"]))
        FN = sum(1 for m in match_list if (m["is_self"] or m["is_similar"]) and not m["is_sig"])
        TN = sum(1 for m in match_list if (not m["is_self"] and not m["is_similar"]) and not m["is_sig"])

    return TP, FP, TN, FN


def compute_precision_recall_f1(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, F1

# =========================================================
# 기존 루프 부분에 기능 추가
# =========================================================
for qlen in sorted(matches_by_qlen.keys(), key=lambda x: int(x)):
    match_list = matches_by_qlen[qlen]

    # ------------------------
    # Confusion Matrix
    # ------------------------
    TP_self, FP_self, TN_self, FN_self = compute_confusion_matrix(match_list, True)
    TP, FP, TN, FN = compute_confusion_matrix(match_list, False)

    # ------------------------
    # Precision / Recall / F1
    # ------------------------
    precision_self, recall_self, F1_self = compute_precision_recall_f1(TP_self, FP_self, FN_self)
    precision, recall, F1 = compute_precision_recall_f1(TP, FP, FN)

    log_and_print(f"[LEN {qlen}] SELF - TP={TP_self} FP={FP_self} TN={TN_self} FN={FN_self}")
    log_and_print(f"[LEN {qlen}] SELF - precision={precision_self:.4f} recall={recall_self:.4f} F1={F1_self:.4f}")
    log_and_print(f"[LEN {qlen}] TP={TP} FP={FP} TN={TN} FN={FN}")
    log_and_print(f"[LEN {qlen}] precision={precision:.4f} recall={recall:.4f} F1={F1:.4f}")

