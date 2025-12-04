# merge_A_B.py
import os
from collections import Counter

A_FILE = "result/folddisco_results_stats/folddisco_result_summary.txt"
B_FILE = "data/folddisco_info_list.txt"
OUT_FILE = "result/folddisco_results_stats/folddisco_result_summary_final.txt"


def extract_pdb_id(path):
    base = os.path.basename(path)
    return base.replace(".pdb", "")

# --- A.txt에서 ID 목록 만들기 ---
a_ids = set()
with open(A_FILE) as f:
    for line in f:
        name = line.strip().split("\t")[0]
        if name.startswith("output_") and name.endswith(".txt"):
            a_ids.add(name[len("output_"):-len(".txt")])

# --- B.txt 읽고, A에 있을 때만 출력 ---
count = 0
with open(B_FILE, "r") as fin, open(OUT_FILE, "w") as fout:
    for line in fin:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        path, value = parts[0], parts[1]
        pdb_id = extract_pdb_id(path)

        if pdb_id in a_ids:
            # 콤마 개수 +1
            comma_count = value.count(",") + 1

            # path 정리
            clean_path = path.replace("data/classified_pdbs/pdbs_", "")

            # A_FILE의 1열 형식 다시 만들기
            # output_{pdb_id}.txt
            a_name = f"output_{pdb_id}.txt"

            # A_FILE처럼 맨 앞에 a_name 붙이고 나머지를 뒤에
            fout.write(f"{a_name}\t{clean_path}\t{value}\t{comma_count}\n")
            count += 1


print(f"[OK] {count} matched entries written to {OUT_FILE}")

