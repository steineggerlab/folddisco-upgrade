#Python script to analyze FoldDisco results, extract max node count and actual PDB length, and summarize results in a text file.

import os

RESULT_DIR = "result/folddisco_results_raw"
STATS_DIR = "result/folddisco_results_stats"
NOKEY_FILE = os.path.join(STATS_DIR, "folddisco_nokey.txt")
RESULT_FILE = os.path.join(STATS_DIR, "folddisco_result_summary.txt")
header = "filename\tlength\tmax_node_count\n"

os.makedirs(STATS_DIR, exist_ok=True)

def get_actual_pdb_length(pdb_id):
    """PDB 파일에서 실제 CA 원자 개수(잔기 수)를 세는 함수"""
    pdb_id = pdb_id.replace("output_", "").split(".txt")[0]
    pdb_path = f"data/index_pdbs/{pdb_id}.pdb"
    
    if not os.path.exists(pdb_path):
        return 0
    
    length = 0
    with open(pdb_path, 'r') as f:
        for line in f:
            # ATOM 레코드이면서 탄소 알파(CA)인 경우만 카운트 (일반적인 단백질 길이 측정 방식)
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                length += 1
    return length

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
                if len(cols) < 3: continue
                
                try:
                    nc = int(cols[1]) # node_count를 숫자로 변환
                    if nc > max_nc:
                        max_nc = nc
                    has_valid_score = True
                except ValueError:
                    continue

            if has_valid_score:
                length = get_actual_pdb_length(filename)
                with open(RESULT_FILE, 'a') as res_file:
                    res_file.write(f"{filename}\t{length}\t{max_nc}\n")
            else:
                with open(NOKEY_FILE, 'a') as n_file:
                    n_file.write(f"{filename}\n")

if __name__ == "__main__":
    main()