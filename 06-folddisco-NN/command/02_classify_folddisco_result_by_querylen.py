#Script to classify Folddisco results by query length and label them based on answer files.

import argparse
import os
import glob
from collections import defaultdict
import re

# 1. 최초 실행 위치의 절대 경로를 칼같이 확보
BASE_DIR = os.path.abspath(os.getcwd())

RESULT_DIR_ORG = "result_benchmark_v2"

# 2. 모든 기준 경로를 최초 위치(BASE_DIR) 기준으로 절대 경로화
RESULT_DIR = os.path.join(BASE_DIR, f"{RESULT_DIR_ORG}")
DATA_DIR = os.path.join(RESULT_DIR, "folddisco_results_raw")
SUMMARY_FILE = f"{RESULT_DIR}/folddisco_results_stats/folddisco_result_summary_final.txt"
RESULT_DIR = f"{RESULT_DIR}/folddisco_results_analyses"
LABEL_NAME = "answer_label" 
benchmark_type = "AFDB"

if benchmark_type == "MCSA":
    ANSWER_DIR = "data/mcsa_new_answers"
elif benchmark_type == "scope40":
    ANSWER_DIR = "data/scope_answers"
else:
    ANSWER_DIR = ""

LOG_FILE = os.path.join(RESULT_DIR, "summary_hits_num.txt")
header_string = "source\tid\tnode_count\tidf_score\trmsd\ttm_score\tmatching_residues\tanswer_label"  # 주의: node_count가 원래 탭 구분과 맞지 않는다면 기존대로 유지

os.makedirs(RESULT_DIR, exist_ok=True)

def log_and_print(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def smart_split(line):
    return line.rstrip("\n").split("\t")


# ----------------------------------------------------
# STEP 1: Load summary mapping (source → length)
# ----------------------------------------------------
length_map = {}

if not os.path.exists(SUMMARY_FILE):
    raise FileNotFoundError(f"Summary file not found: {SUMMARY_FILE}")

with open(SUMMARY_FILE) as f:
    for line in f:
        if line.startswith("filename") or not line.strip():
            continue
        parts = line.split()
        source = parts[0]          # ex: output_1beaA00.txt 또는 scope 디렉토리명/파일명
        length = int(parts[1])
        length_map[source] = length


# ----------------------------------------------------
# STEP 2: Process each output file in raw results
# ----------------------------------------------------
for source, length in length_map.items():
    # scope40의 경우 source가 디렉토리명처럼 동작하거나, 패턴 매칭이 필요할 수 있으므로 recursive 유지
    pattern = os.path.join(DATA_DIR, "**", source)
    matches = glob.glob(pattern, recursive=True)

    if not matches:
        continue  

    raw_file = matches[0]
    rel_source = os.path.relpath(raw_file, DATA_DIR)
    combined_path = os.path.join(RESULT_DIR, f"length_{length}_combined.txt")
    write_header = not os.path.exists(combined_path)

    with open(combined_path, "a") as out:
        if write_header:
            out.write(f"{header_string}\n")

        with open(raw_file) as rf:
            lines = rf.read().splitlines()

        if len(lines) <= 1:
            continue

        data_lines = lines[1:]  # 원본 헤더 제외

        for row in data_lines:
            row = row.strip()
            if row:
                out.write(f"{source}\t{row}\n")

print(f"Step 1-2 completed: Files processed and combined by query length in {RESULT_DIR}")


# ----------------------------------------------------
# STEP 2.5: [NEW] scope40 파일 경로 사전 캐싱 (Glob 병목 제거)
# ----------------------------------------------------
raw_file_map = {}
if benchmark_type == "scope40":
    print("스크립트 속도 최적화를 위해 디렉토리 구조를 사전 캐싱 중입니다...")
    # DATA_DIR 내의 모든 .txt 파일을 한 번에 스캔
    for root, dirs, files_list in os.walk(DATA_DIR):
        for f in files_list:
            if f.endswith(".txt"):
                # 예: {'d1ngka__0.8_1.txt': 'a.1.1.1'}
                raw_file_map[f] = os.path.basename(root).strip()
    print(f"총 {len(raw_file_map)}개의 원본 결과 파일 경로를 캐싱했습니다.")

# ----------------------------------------------------
# STEP 3: Load Answer Dictionary (MCSA vs scope40 분기)
# ----------------------------------------------------
ANSWER_DICT = defaultdict(set)  # list 대신 set 사용!

if benchmark_type == "AFDB":
    print("[STEP 3 skip] AFDB는 정답 파일 없이 query==target 매칭으로 라벨링합니다.")
else:
    for file in os.listdir(ANSWER_DIR):
        if benchmark_type == "MCSA":
            if not file.endswith("_answer.tsv"):
                continue
            file_name = file.replace("_answer.tsv", "").strip()

            answer_path = os.path.join(ANSWER_DIR, file)
            with open(answer_path, "r", encoding="utf-8", errors="ignore") as answer:
                for line in answer:
                    if line.strip():
                        ANSWER_DICT[file_name].add(line.strip())  # set.add()

        else:  # scope40
            if not file.endswith(".txt") or file.startswith("."):
                continue

            file_name = file.split(".txt")[0].strip()
            answer_path = os.path.join(ANSWER_DIR, file)

            with open(answer_path, "r", encoding="utf-8", errors="ignore") as answer:
                for line in answer:
                    clean_line = line.strip()
                    if not clean_line:
                        continue

                    cols = clean_line.split()
                    if cols:
                        domain_id = cols[0].strip().rstrip('_')
                        ANSWER_DICT[file_name].add(domain_id)

    print(f"[STEP 3 완료] 총 {len(ANSWER_DICT)}개의 정답 세트를 셋(Set) 구조로 로드했습니다.")
# ----------------------------------------------------
# STEP 4: Labeling and Filtering
# ----------------------------------------------------
delete_file = []

files = [
    os.path.join(RESULT_DIR, f) 
    for f in os.listdir(RESULT_DIR) 
    if f.startswith("length_") and f.endswith(".txt")
]

for file_path in files:
    rows = []
    p_answer = []
    
    # 대형 파일 진행 상황 확인용
    line_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        print(f"Processing: {file_path}")
        for line in f:
            cols = smart_split(line)
            if len(cols) == 0 or (len(cols) == 1 and cols[0] == ''):
                continue
            rows.append(cols)
    
    if not rows:
        continue

    header = rows[0]
    start_i = 0
    if header[0] == "source":
        if LABEL_NAME not in header:
            header.append(LABEL_NAME)
        rows[0] = header
        start_i = 1

    # 루프 돌기 전 필요한 참조 함수나 내장 메서드를 로컬 변수화하여 파이썬 가속
    if benchmark_type == "MCSA":
        for i in range(start_i, len(rows)):
            cols = rows[i]
            if len(cols) < 2: continue
            
            q = cols[0].replace(".txt", "")
            t = cols[1].split("/")[-1].strip().replace(".pdb", "")
            
            if q in ANSWER_DICT and t in ANSWER_DICT[q]: # O(1)
                label = 1
                p_answer.append(q)
            else:
                label = 0
            cols.append(str(label))

    elif benchmark_type == "AFDB":
        for i in range(start_i, len(rows)):
            cols = rows[i]
            if len(cols) < 2: continue

            # Regex로 'AF-XXXXXX-FX-model_vX' 패턴만 추출
            q_match = re.search(r'(AF-[A-Za-z0-9]+-F\d+-model_v\d+)', cols[0])
            t_match = re.search(r'(AF-[A-Za-z0-9]+-F\d+-model_v\d+)', cols[1])

            q = q_match.group(1) if q_match else cols[0]
            t = t_match.group(1) if t_match else cols[1]

            if q == t:
                label = 1
                p_answer.append(q)
            else:
                label = 0
            cols.append(str(label))
    
    else:  # scope40 (1,500만 행 루프 고속화 부품)
        for i in range(start_i, len(rows)):
            cols = rows[i]
            if len(cols) < 2: continue
            
            key = cols[0]
            target = cols[1]

            # 1) 디스크 조회 없이 메모리 딕셔너리 룩업으로 q 추출 ($O(1)$)
            raw_filename = key if key.endswith(".txt") else f"{key}.txt"
            q = raw_file_map.get(raw_filename, "")
            if not q:
                q = os.path.dirname(key).strip() if "/" in key else key.replace(".txt", "").strip()

            # 2) Target 파일 ID 정형화
            t = target.split("/")[-1].strip().replace(".ent", "").rstrip('_')

            # 3) 매칭 및 라벨링 ($O(1)$ 해시 매칭)
            # 이제 ANSWER_DICT[q]가 set이므로 in 연산이 즉시 끝납니다.
            if q in ANSWER_DICT and t in ANSWER_DICT[q]:
                label = 1
                p_answer.append(q)
            else:
                label = 0

            cols.append(str(label))

    # 파일 쓰기
    with open(file_path, "w", encoding="utf-8") as out:
        for cols in rows:
            out.write("\t".join(cols) + "\n")

    if len(p_answer) < 10:
        log_and_print(f'File passed\t{file_path}\t{len(p_answer)}')
        delete_file.append(file_path)
    else:
        log_and_print(f'File added\t{file_path}\t{len(p_answer)}')

# 삭제 수행
for fp in delete_file:
    log_and_print(f'Removing: {fp}')
    if os.path.exists(fp):
        os.remove(fp)

