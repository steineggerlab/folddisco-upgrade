#Script to gather Folddisco results by query length

import os
import glob
import argparse
# 1. 최초 실행 위치의 절대 경로를 칼같이 확보
BASE_DIR = os.path.abspath(os.getcwd())

RESULT_DIR_ORG = "result_benchmark_v2"

# 2. 모든 기준 경로를 최초 위치(BASE_DIR) 기준으로 절대 경로화
RESULT_DIR = os.path.join(BASE_DIR, f"{RESULT_DIR_ORG}")

TARGET_DIR = f"{RESULT_DIR}/folddisco_results_analyses"
FINAL_FILE = os.path.join(TARGET_DIR, "total_combined.txt")

pattern = os.path.join(TARGET_DIR, "length_*_combined.txt")
files = sorted(glob.glob(pattern))

if not files:
    raise SystemExit(f"No length_*_combined.txt found in: {TARGET_DIR}")

# Read header from first file
with open(files[0], "r") as f:
    header_line = f.readline().rstrip("\n")

if not header_line:
    raise SystemExit(f"First file has empty header: {files[0]}")

os.makedirs(TARGET_DIR, exist_ok=True)

with open(FINAL_FILE, "w") as out:
    # Unified header
    out.write("query_num\t" + header_line + "\n")

    # Append rows from each file
    for fp in files:
        qnum = os.path.basename(fp)
        with open(fp, "r") as f:
            # skip per-file header
            _ = f.readline()
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                out.write(f"{qnum}\t{line}\n")

print(f"[OK] Created: {FINAL_FILE}")