#Python script to analyze FoldDisco results, extract max node count and actual PDB length, and summarize results in a text file.
import argparse
import os

RESULT_DIR_ORG = "result_benchmark_v2"

RESULT_DIR = os.path.join(RESULT_DIR_ORG, "folddisco_results_raw")
STATS_DIR = os.path.join(RESULT_DIR_ORG, "folddisco_results_stats")
NOKEY_FILE = os.path.join(STATS_DIR, "folddisco_nokey.txt")
RESULT_FILE = os.path.join(STATS_DIR, "folddisco_result_summary_final.txt")
header = "filename\tmax_node_count\n"

os.makedirs(STATS_DIR, exist_ok=True)

def main():
    print("Starting analysis...")
    
    # 헤더 작성
    with open(RESULT_FILE, 'w') as res_file:
        res_file.write(header)

    for root, _, files in os.walk(RESULT_DIR):
        for filename in files:
            if not filename.endswith(".txt"):
                continue
                
            file_path = os.path.join(root, filename)
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            # 데이터가 없는 경우 (헤더 제외 1줄 이하)
            if len(lines) <= 1:
                with open(NOKEY_FILE, 'a') as n_file:
                    n_file.write(f"{filename}\n")
                continue

            max_nc = -1
            has_valid_score = False

            for line in lines[1:]: # 헤더 건너뜀
                cols = line.strip().split("\t")
                if len(cols) < 5: continue
                
                try:
                    nc = int(cols[1]) # node_count를 숫자로 변환
                    if nc > max_nc:
                        max_nc = nc
                    has_valid_score = True
                except ValueError:
                    continue

            if has_valid_score:
                with open(RESULT_FILE, 'a') as res_file:
                    res_file.write(f"{filename}\t{max_nc}\n")
            else:
                with open(NOKEY_FILE, 'a') as n_file:
                    n_file.write(f"{filename}\n")
    print("Analysis completed.")

if __name__ == "__main__":
    main()
