#Python script to analyze Folddisco e-value results by query length (especially significant & self hits)

import os
from collections import defaultdict

DATA_DIR = "result/folddisco_result_2_revisited"
DATA_FILE = os.path.join(DATA_DIR, "total_evalues_fitted2.txt") 
RESULT_FILE = os.path.join(DATA_DIR, "evalues_stat2.txt") 
LOG_FILE = os.path.join(DATA_DIR, "summary_log2.txt")
EVAL_CUTOFF = 10


def log_and_print(message):
    print(message)
    with open(LOG_FILE, "a") as log:
        log.write(message + "\n")


length_allqueries = defaultdict(list)
length_sigqueries = defaultdict(set)
length_selfhits   = defaultdict(set)


with open(DATA_FILE, "r") as f:
    lines = f.read().splitlines()

header = lines[0].strip()
filtered_rows = []

for line in lines[1:]:
    cols = line.split("\t")
    if len(cols) < 6:
        continue
    qlen  = cols[0] 
    query = cols[1]
    index  = cols[2]
    e_value = cols[5]

    length_allqueries[qlen].append(query)

    try:
        e_val_f = float(e_value)
    except ValueError:
        continue

    if e_val_f < EVAL_CUTOFF:
        filtered_rows.append(line)
        length_sigqueries[qlen].add(query)

        q_short = query.replace("output_", "").replace(".txt", "")
        idx_short = index.replace("data/index_pdbs/", "").replace(".pdb", "")
        if q_short == idx_short:
            length_selfhits[qlen].add(query)

with open(RESULT_FILE, "w") as out:
    out.write(header + "\n")
    for line in filtered_rows:
        out.write(line + "\n")

log_and_print(f"[OK] Saved {len(filtered_rows)} significant hits (e<{EVAL_CUTOFF}) -> {RESULT_FILE}")

log_and_print(f"=== Summary by query length (e<{EVAL_CUTOFF}) ===")

for qlen in sorted(length_allqueries.keys(), key=lambda x: int(x.split("_")[0])):
    total_queries = len(length_allqueries[qlen])
    n_sig  = len(length_sigqueries.get(qlen, []))
    n_self = len(length_selfhits.get(qlen, []))
    log_and_print(f"{qlen}: {n_sig} significant / {n_self} self-hits / {total_queries} total queries")

log_and_print("========================================")