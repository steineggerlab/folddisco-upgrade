RESULT_YES_FILE = "result/folddisco_result_2_stat/folddisco_result_summary.txt"
RESULT_NO_FILE = "result/folddisco_result_2_stat/folddisco_nokey.txt"
DATA_DIR = 'data/classified_pdbs'
CLASSIFIED_FILE = 'result/folddisco_result_2_stat/folddisco_result_yesorno.txt'
QUERY_INFO = 'domain_list.txt'
DIRECTORY_FILE = 'result/folddisco_result_2_stat/directory_result.tsv'
QUERY_DIRECTORY='data/query_sequences_1'
DIFFERENCE_RESULT = 'result/folddisco_result_2_stat/difference.tsv'

import os, collections, ast

def to_pdb_id(key: str) -> str:
    base = os.path.splitext(os.path.basename(key))[0]  
    if base.startswith("output_"):
        base = base[len("output_"):]                   
    if base.startswith("pdbs_"):
        base = base[len("pdbs_"):]                    
    return base

with open(RESULT_YES_FILE) as f:
    yes_ids = {line.strip().split()[0] for line in f if line.strip()}

with open(RESULT_NO_FILE) as f:
    no_ids = {line.strip().split()[0] for line in f if line.strip()}

with open(CLASSIFIED_FILE, "w") as classified_file:
    for root, dirs, files in os.walk(DATA_DIR):
        if QUERY_INFO in files:
            cat_id = os.path.basename(root)
            for pdb in [f for f in os.listdir(root) if f.endswith(".pdb")]:
                pdb_id = os.path.splitext(pdb)[0]
                pdb_id = 'output_' + pdb_id + '.txt'
                if pdb_id in yes_ids:
                    classified_file.write(f"{pdb_id}\t{root}\tYES\n")
                elif pdb_id in no_ids:
                    classified_file.write(f"{pdb_id}\t{root}\tNO\n")
                else:
                    classified_file.write(f"{pdb_id}\t{root}\n")

stats = collections.defaultdict(lambda: {"YES": 0, "NO": 0, "UNCLASSIFIED": 0, "RATIO": 0.0})

with open(CLASSIFIED_FILE) as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        pdb_id, path = parts[0], parts[1]
        status = parts[2] if len(parts) > 2 else "UNCLASSIFIED"
        stats[path][status] += 1

sorted_stats = sorted(stats.items(), key=lambda x: x[1]["NO"], reverse=True)
no_existing_directory = {}

with open(DIRECTORY_FILE, "w") as out:
    out.write("directory\tYES\tNO\tUNCLASSIFIED\n")
    for path, counts in sorted_stats:
        yes, no = counts["YES"], counts["NO"]
        denom = yes + no
        ratio = yes / denom if denom > 0 else 0.0
        if ratio < 1.0:
            no_existing_directory[path] = ratio
        out.write(f"{path}\t{yes}\t{no}\t{counts['UNCLASSIFIED']}\t{ratio:.3f}\n")

with open(DIFFERENCE_RESULT, "w") as out:
    out.write("pdb\t\tminimum_foldmason\treal_ratio\n")
    for pdb, thr in no_existing_directory.items():
        pdb_id = to_pdb_id(pdb)
        qpath = os.path.join(QUERY_DIRECTORY, f"{pdb_id}_query.txt")
        try:
            last_line = open(qpath).read().strip().splitlines()[-1]
            min_ratio = min(map(float, ast.literal_eval(last_line)))
            if min_ratio > float(thr):
                out.write(f"{pdb}\t{min_ratio:.3f}\t{float(thr):.3f}\n")
        except Exception:
            pass



print("[OK] Saved:", DIRECTORY_FILE)
