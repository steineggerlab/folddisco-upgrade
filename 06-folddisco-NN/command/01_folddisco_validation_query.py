# Python script to validate the query information for Folddisco and execute the Folddisco command.

import argparse
import os, subprocess
from collections import defaultdict
from Bio.PDB import PDBParser
import numpy as np
import itertools
import re

# 1. 최초 실행 위치의 절대 경로를 칼같이 확보
BASE_DIR = os.path.abspath(os.getcwd())

RESULT_DIR_ORG = "result_benchmark_v2"

# 2. 모든 기준 경로를 최초 위치(BASE_DIR) 기준으로 절대 경로화
RESULT_DIR = os.path.join(BASE_DIR, f"{RESULT_DIR_ORG}")
OUTPUT_DIR = os.path.join(RESULT_DIR, "folddisco_results_raw")
benchmark_type = "AFDB"

if benchmark_type == "MCSA":
    INDEX_DIR = '/fast2/hyunbin/folddisco_revision/250721_folddisco_querying_benchmark/index/mcsa_subset_folddisco'
    INDEX_PDB_DIR = '/fast2/hyunbin/folddisco_revision/250721_folddisco_querying_benchmark/index/mcsa_subset'
    INFO_LIST_FILE = os.path.join(BASE_DIR, 'data/data_pos_mcsa/folddisco_info_list_MCSA.txt')
elif benchmark_type == "scope40":
    INDEX_DIR = '/fast2/hyunbin/folddisco_revision/250721_folddisco_querying_benchmark/index/scope40_folddisco'
    INDEX_PDB_DIR = '/fast2/hyunbin/folddisco_revision/250721_folddisco_querying_benchmark/index/scope40/pdbstyle-2.08'
    INFO_LIST_FILE = os.path.join(BASE_DIR, 'data/data_bench_scope40/folddisco_info_list_scope40.txt')
else:
    INDEX_DIR = '/fast2/jwyoon05/folddisco-dirs/folddisco-NN-afdbshuffle/index/afdb_shuffled_folddisco'
    INDEX_PDB_DIR = '/fast2/jwyoon05/folddisco-dirs/folddisco-NN-afdbshuffle/data/afdb_shuffled'
    INFO_LIST_FILE = os.path.join(BASE_DIR, 'data/data_neg_afdb_shuffle/folddisco_info_list_expanded.txt')

PDB_FILE = os.path.join(BASE_DIR, "data/pdb_progress.txt")
FOLDDISCO_COMMAND_FILE = os.path.join(BASE_DIR, "command/folddisco_commands.txt")
WEIRD_FILE = os.path.join(BASE_DIR, f'data/weird_pdbs_{benchmark_type}.txt')

# 3. 디렉토리 이동 (이동 후에도 위 절대 경로들은 영향을 받지 않음)
if os.path.exists("folddisco"): 
    os.chdir("folddisco")

# folddisco 폴더 진입 후 바이너리 위치 확정
FOLDDISCO = "./folddisco" if os.path.exists("./folddisco") else "folddisco"
DISTANCE_CUTOFF = 20.0

# 4. 절대 경로 기반으로 안전하게 디렉토리 및 파일 초기화
os.makedirs(os.path.dirname(WEIRD_FILE), exist_ok=True)
os.makedirs(os.path.dirname(PDB_FILE), exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

open(WEIRD_FILE, 'w').close()
open(PDB_FILE, 'w').close()

weird_pdbs = {}

if not os.path.exists(INFO_LIST_FILE):
    print(f"[ERROR] INFO_LIST_FILE does not exist: {INFO_LIST_FILE}")
    exit(1)

parser = PDBParser(QUIET=True)

def ca_distance(res1, res2):
    try:
        return np.linalg.norm(res1["CA"].coord - res2["CA"].coord)
    except KeyError:
        return None

# ---------------------------------------------------
# Main loop
# ---------------------------------------------------
print(f"Validating existing entries in: {INFO_LIST_FILE}")

with open(INFO_LIST_FILE, "r") as infile:
    for line in infile:
        if not line.strip():
            continue
            
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) < 3:
            continue
            
        pdb_path, residue_str, _ = parts[0], parts[1], parts[2]
        pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
        residues = residue_str.split(",")

        with open(PDB_FILE, 'a') as pf:
            pf.write(f"{pdb_id}\n")

        # Load PDB
        try:
            structure = parser.get_structure("X", pdb_path)
        except:
            weird_pdbs[pdb_id] = "PDBLoadError"
            continue

        try:
            model = next(iter(structure))
        except StopIteration:
            weird_pdbs[pdb_id] = "EmptyPDBStructure"
            continue

        # (2) Residue 존재 여부 확인
        coords = []
        all_found = True

        for r in residues:
            m = re.match(r"([A-Za-z0-9])(\d+)([A-Za-z]?)", r.strip())
            if not m:
                all_found = False
                weird_pdbs[pdb_id] = "InvalidResidueFormat"
                break
            
            if m.group(3) != "":
                weird_pdbs[pdb_id] = "InsertionCodeNotSupported"
                all_found = False
                break

            chain_id = m.group(1)
            resseq = int(m.group(2))

            if chain_id not in model:
                all_found = False
                weird_pdbs[pdb_id] = "MissingChain"
                break

            chain = model[chain_id]
            found = False

            for res in chain:
                het, seq, ins = res.get_id()
                if seq == resseq and het.strip() == "":
                    coords.append(res)
                    found = True
                    break

            if not found:
                all_found = False
                weird_pdbs[pdb_id] = "ResidueNotFound"
                break

        if not all_found:
            continue

        if len(coords) < 2:
            weird_pdbs[pdb_id] = "InsufficientResidues"
            continue

        # (3) Distance cutoff
        if len(coords) >= 2:
            all_distant = True
            for a, b in itertools.combinations(coords, 2):
                d = ca_distance(a, b)
                if d is not None and d <= DISTANCE_CUTOFF:
                    all_distant = False
                    break

            if all_distant:
                weird_pdbs[pdb_id] = "AllResiduesDistant"
                continue

# ----------------------------
# Write weird cases
# ----------------------------
with open(WEIRD_FILE, 'w') as wf:
    for pdb_id, reason in weird_pdbs.items():
        wf.write(f"{reason}\t{pdb_id}\n")

# ---------------------------------------------------
# [수정] 현재 위치(folddisco/) 기준에 정확히 심볼릭 링크 생성
# ---------------------------------------------------
if benchmark_type == "MCSA":
    os.makedirs("index", exist_ok=True)
    symlink_target = "index/mcsa_subset"
    
    if os.path.exists(symlink_target) or os.path.islink(symlink_target):
        if os.path.islink(symlink_target):
            os.unlink(symlink_target)
        else:
            import shutil
            shutil.rmtree(symlink_target)
            
    print(f"[LINK] Creating MCSA symbolic link: {symlink_target} -> {INDEX_PDB_DIR}")
    os.symlink(INDEX_PDB_DIR, symlink_target)

elif benchmark_type == "scope40":
    # 1. 러스트 엔진이 요구하는 index/scope40 부모 폴더를 먼저 생성
    os.makedirs("index/scope40", exist_ok=True)
    symlink_target = "index/scope40/pdbstyle-2.08"
    
    if os.path.exists(symlink_target) or os.path.islink(symlink_target):
        if os.path.islink(symlink_target):
            os.unlink(symlink_target)
        else:
            import shutil
            shutil.rmtree(symlink_target)
            
    # 2. index/scope40/pdbstyle-2.08 지름길을 진짜 2블록 하위 디렉토리 셋팅에 링크
    print(f"[LINK] Creating scope40 symbolic link: {symlink_target} -> {INDEX_PDB_DIR}")
    os.symlink(INDEX_PDB_DIR, symlink_target)

else:
    os.makedirs("index/afdb_shuffled", exist_ok=True)
    symlink_target = "index/afdb_shuffled/data"
    
    if os.path.exists(symlink_target) or os.path.islink(symlink_target):
        if os.path.islink(symlink_target):
            os.unlink(symlink_target)
        else:
            import shutil
            shutil.rmtree(symlink_target)
            
    print(f"[LINK] Creating AFDB symbolic link: {symlink_target} -> {INDEX_PDB_DIR}")
    os.symlink(INDEX_PDB_DIR, symlink_target)

# 명령어 파일 저장 및 실행
os.makedirs(os.path.dirname(FOLDDISCO_COMMAND_FILE), exist_ok=True)
command = (
    f"{FOLDDISCO} query -i {INDEX_DIR} "
    f"-q {INFO_LIST_FILE} -t 32 -v --sort-by idf --top 10000 --header --format-output tid,node_count,idf,rmsd,tm_score,matching_residues"
)
with open(FOLDDISCO_COMMAND_FILE, 'w') as f:
    f.write(command + "\n")

print("Executing folddisco commands.")
os.system(f"bash {FOLDDISCO_COMMAND_FILE}")
print("Folddisco finished.")
