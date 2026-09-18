import os

# 파일 경로 설정 (본인 환경에 맞게 수정 가능)
INPUT_FILE = "data/scope40_folddisco_per_structure_260112.txt"
OUTPUT_FILE = "data/folddisco_info_list_scope40.txt"

# 요청하신 정확한 절대경로 접두사 정의 (끝에 '/' 포함)
PREFIX_1ST_COL = "/fast2/hyunbin/folddisco_revision/250721_folddisco_querying_benchmark/"
PREFIX_3RD_COL = "/fast2/jwyoon05/folddisco-dirs/folddisco-NN/result_benchmark_scope40/folddisco_results_raw/"

print("Starting file reformatting...")

with open(INPUT_FILE, "r") as infile, open(OUTPUT_FILE, "w") as outfile:
    for line in infile:
        if not line.strip():
            continue
            
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
            
        col1, col2, col3 = parts[0], parts[1], parts[2]
        
        # 1열: 지정된 절대경로 접두사 결합
        new_col1 = f"{PREFIX_1ST_COL}{col1}"
        
        # 2열: 그대로 유지
        new_col2 = col2
        
        # 3열: 기존 3열 파일명에서 'M0001' 같은 ID만 추출 후, 지정하신 절대경로 접두사 + .txt 결합
        base_name = os.path.basename(col3)
        file_id, _ = os.path.splitext(base_name)  # "M0001", ".tsv" 분리

        parent_dir = os.path.dirname(col3)
        dir_name = os.path.basename(parent_dir)
        new_col3 = f"{PREFIX_3RD_COL}{dir_name}/{file_id}.txt"
        
        # 탭 구분으로 저장
        outfile.write(f"{new_col1}\t{new_col2}\t{new_col3}\n")

print(f"Reformatting completed! Saved to '{OUTPUT_FILE}'.")