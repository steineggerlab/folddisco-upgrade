# Python script to merge_A and B files based on extracted IDs and write the final output
import os

# 파일 경로 설정
A_FILE = "result/folddisco_results_stats/folddisco_result_summary.txt"
B_FILE = "data/folddisco_info_list.txt"
OUT_FILE = "result/folddisco_results_stats/folddisco_result_summary_final.txt"

def get_core_id_from_filename(filename):
    """
    파일명에서 핵심 ID만 추출 (숫자 꼬리표 제거)
    output_1wvfA02.txt3114.txt -> 1wvfA02
    """
    return filename.replace("output_", "").split(".txt")[0]

def extract_pdb_id_from_path(path):
    """
    파일 경로에서 ID 추출
    data/.../1wvfA02.pdb -> 1wvfA02
    """
    base = os.path.basename(path)
    return base.replace(".pdb", "")

# ----------------------------------------------------
# 1. A 파일(실제 결과 파일 목록)을 먼저 읽어서 맵핑 생성
#    { '핵심ID' : '실제파일명' } 구조로 저장
#    예: { '1wvfA02': 'output_1wvfA02.txt3114.txt' }
# ----------------------------------------------------
id_to_realname = {}

print(f"Loading results from {A_FILE}...")
with open(A_FILE, 'r') as f:
    for line in f:
        parts = line.strip().split("\t")
        if not parts:
            continue
            
        real_filename = parts[0]  # output_1wvfA02.txt3114.txt
        
        # 파일명 형식이 맞는지 확인
        if real_filename.startswith("output_") and ".txt" in real_filename:
            core_id = get_core_id_from_filename(real_filename)
            id_to_realname[core_id] = real_filename

print(f"Loaded {len(id_to_realname)} result files.")

# ----------------------------------------------------
# 2. B 파일을 순회하며 매칭 및 최종 파일 작성
# ----------------------------------------------------
count = 0
with open(B_FILE, "r") as fin, open(OUT_FILE, "w") as fout:
    for line in fin:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        
        path_b = parts[0]   # 예: data/classified_pdbs/.../1wvfA02.pdb
        value = parts[1]    # 예: 1,2,3
        
        # B 파일의 경로에서 ID 추출
        pdb_id_b = extract_pdb_id_from_path(path_b)

        # 추출한 ID가 A 파일(실제 결과) 목록에 있는지 확인
        if pdb_id_b in id_to_realname:
            # A 파일에 기록되어 있던 '진짜 파일명' (숫자가 붙은 것)을 가져옴
            final_filename = id_to_realname[pdb_id_b]
            
            # 기타 데이터 가공
            comma_count = value.count(",") + 1
            clean_path = path_b.replace("data/classified_pdbs/pdbs_", "")

            # 최종 기록: 첫 번째 컬럼에 '진짜 파일명'을 적음
            fout.write(f"{final_filename}\t{clean_path}\t{value}\t{comma_count}\n")
            count += 1

print(f"[OK] {count} matched entries written to {OUT_FILE}")