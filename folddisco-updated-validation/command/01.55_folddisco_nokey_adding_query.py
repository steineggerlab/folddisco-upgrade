# merge_A_B.py
import os

A_FILE = "result/folddisco_results_stats/folddisco_nokey_only_for_old.txt"
B_FILE = "data/folddisco_info_list.txt"
OUT_FILE = "result/folddisco_results_stats/folddisco_nokey_merged_only_for_old.txt"


def extract_pdb_id(path):
    """data/classified_pdbs/.../3b0zB00.pdb → 3b0zB00"""
    base = os.path.basename(path)
    return base.replace(".pdb", "")

# --- A.txt에서 ID 목록 만들기 ---
a_ids = set()
with open(A_FILE) as f:
    for line in f:
        name = line.strip()
        if name.startswith("output_") and name.endswith(".txt"):
            a_ids.add(name[len("output_"):-len(".txt")])

# --- B.txt 읽고, A에 있을 때만 출력 ---
count = 0
with open(B_FILE) as fin, open(OUT_FILE, "w") as fout:
    for line in fin:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        path, value = parts[0], parts[1]
        pdb_id = extract_pdb_id(path)

        if pdb_id in a_ids:
            fout.write(f"{path}\t{value}\n")
            count += 1

print(f"[OK] {count} matched entries written to {OUT_FILE}")