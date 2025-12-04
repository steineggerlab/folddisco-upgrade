import os
import glob

DATA_DIR = "result/folddisco_results_raw"
SUMMARY_FILE = "result/folddisco_results_stats/folddisco_result_summary_final.txt"
RESULT_DIR = "result/folddisco_results_analyses"
PDB_FILE = "data/domain-list-index.txt"

LOG = f"summary_hits_num.txt"
LOG_FILE = os.path.join(RESULT_DIR, LOG)
header = (
    "source\tid\tnode_count\tidf_score_per_match\trmsd\tmatching_residues\tkey\t"
    "tm_score\ttm_score_strict\tgdt_ts\tgdt_ha\tgdt_strict\trmsd\tchamfer_distance\t"
    "hausdorff_distance\tquery_residues"
)

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

with open(SUMMARY_FILE) as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue

        parts = line.split()
        if len(parts) < 4:
            continue

        source = parts[0]          # ex: output_1beaA00.txt
        length = int(parts[3])    

        length_map[source] = length


# ----------------------------------------------------
# STEP 2: Process each output file in raw results
# ----------------------------------------------------
for source, length in length_map.items():

    # output raw file 위치 찾기
    pattern = os.path.join(DATA_DIR, "**", source)
    matches = glob.glob(pattern, recursive=True)

    if not matches:
        continue  # raw output 파일 없는 경우 skip

    raw_file = matches[0]

    # length_X_combined 파일 준비
    combined_path = os.path.join(RESULT_DIR, f"length_{length}_combined.txt")

    # 파일이 새로 생성되는 경우 header 작성
    write_header = not os.path.exists(combined_path)

    with open(combined_path, "a") as out:
        if write_header:
            out.write(header + "\n")

        # ----------------------------------------------------
        # STEP 2-1: raw output 파일에서 data rows 가져오기
        # ----------------------------------------------------
        with open(raw_file) as rf:
            lines = rf.read().splitlines()

        if len(lines) <= 1:
            continue

        data_lines = lines[1:]  # 첫 줄은 원본 header 제거

        # ----------------------------------------------------
        # STEP 2-2: 모든 row를 그대로 length_X 파일에 쓰기
        # ----------------------------------------------------
        for row in data_lines:
            row = row.strip()
            if row:
                out.write(f"{source}\t{row}\n")


SUBGROUP_DICT = {}

def subgroup_detector(line):
    cols = smart_split(line)
    if len(cols) < 2:
        return None

    pdb_id = cols[0]
    subgroup = cols[1] + cols[2] + cols[3]

    SUBGROUP_DICT[pdb_id] = subgroup
    return None

with open(PDB_FILE, 'r') as data:
    for data_line in data.readlines():
        subgroup_detector(data_line)


delete_file = []
files = [os.path.join(RESULT_DIR, f) for f in os.listdir(RESULT_DIR)]

for file_path in files:
    rows = []
    p_self = []
    p_similar = []

    with open(file_path) as f:
        print(file_path)
        for line in f:
            cols = smart_split(line)
            if len(cols) == 0:
                continue
            rows.append(cols)

    # 각 row에서 self / similar 판정
    for cols in rows:
        key = cols[0]
        target = cols[1]

        q = key.replace("output_", "").replace(".txt", "")
        t = os.path.basename(target).replace(".pdb", "")

        if q == t:
            p_self.append(q)
        elif SUBGROUP_DICT.get(q, "") == SUBGROUP_DICT.get(t, "") and SUBGROUP_DICT.get(q, "") != "":
            p_similar.append(q)

    # 개수 체크
    if len(p_self) < 30 or len(p_similar) < 30:
        log_and_print(f'File added\t{file_path}\t{len(p_self)}\t{len(p_similar)}')
        delete_file.append(file_path)
    
    else:
        log_and_print(f'File passed{file_path}')

# 삭제 수행
for fp in delete_file:
    log_and_print(f'Removing:{fp}')
    os.remove(fp)