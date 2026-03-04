#Python script to validate the query information for Folddisco and execute the Folddisco command.

import os, subprocess
from collections import defaultdict
from Bio.PDB import PDBParser
import numpy as np
import itertools
import re

if os.path.exists("folddisco"): 
    os.chdir("folddisco")

DATA_DIR='../data/classified_pdbs'
DOMAIN_INFO='domain_list.txt'
QUERY_INFO='../data/chain_residue_list.txt'
INDEX_DIR='index/index_pdbs_folddisco' 
RESULT_DIR='../result/folddisco_results_raw'
INFO_LIST_FILE='../data/folddisco_info_list.txt'
FOLDDISCO_COMMAND_FILE="../command/folddisco_commands.txt"
WEIRD_FILE='../data/weird_pdbs.txt'
FOLDDISCO="folddisco"
BIGNUM=999999999999
DISTANCE_CUTOFF=20.0

os.makedirs(RESULT_DIR, exist_ok=True)

parser = PDBParser(QUIET=True)

def ca_distance(res1, res2):
    if "CA" not in res1 or "CA" not in res2:
        return None
    return np.linalg.norm(res1["CA"].coord - res2["CA"].coord)

# ---------------------------
# Load QUERY_INFO completely
# ---------------------------
query_map = {}
with open(QUERY_INFO) as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            folder, pid, res = parts[0], parts[1], parts[2].split(",")
            query_map[(folder, pid)] = res
        elif len(parts) == 2:
            folder, pid = parts[0], parts[1]
            query_map[(folder, pid)] = []
        else:
            continue

# initialize output files
open(INFO_LIST_FILE, 'w').close()
open(WEIRD_FILE, 'w').close()

weird_pdbs = {}  # pdb_id → reason

# ---------------------------------------------------
# Main loop
# ---------------------------------------------------
for file in os.listdir(DATA_DIR):

    if DOMAIN_INFO not in os.listdir(os.path.join(DATA_DIR, file)):
        continue

    dir_path = os.path.join(DATA_DIR, file)
    relative_path = os.path.relpath(dir_path, DATA_DIR)

    for pdb in os.listdir(dir_path):
        if not pdb.endswith('.pdb'):
            continue

        pdb_id = pdb.replace('.pdb', '')
        pdb_path = os.path.join(dir_path, pdb)

        # (1) Query mapping 존재 여부 확인
        if (relative_path, pdb_id) not in query_map:
            weird_pdbs[pdb_id] = "MissingQueryEntry"
            continue

        residues = query_map[(relative_path, pdb_id)]

        # Load PDB
        try:
            structure = parser.get_structure("X", pdb_path)
        except:
            weird_pdbs[pdb_id] = "PDBLoadError"
            continue

        model = structure[0]

        # (2) Residue 존재 여부 확인
        coords = []
        all_found = True

        for r in residues:
            m = re.match(r"([A-Za-z])(\d+)([A-Za-z]?)", r)
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
                if seq == resseq:
                    coords.append(res)
                    found = True
                    break

            if not found:
                all_found = False
                weird_pdbs[pdb_id] = "ResidueNotFound"
                break

        # 일부 residue라도 실패하면 skip + weird 기록
        if not all_found:
            continue

        if len(coords) < 2:
            weird_pdbs[pdb_id] = "InsufficientResidues"
            continue

        # (3) Distance cutoff (residue ≥ 2 일 때만)
        if len(coords) >= 2:
            dlist = []
            for a, b in itertools.combinations(coords, 2):
                d = ca_distance(a, b)
                if d is not None:
                    dlist.append(d)

            if len(dlist) > 0 and min(dlist) > DISTANCE_CUTOFF:
                weird_pdbs[pdb_id] = "AllResiduesDistant"
                continue

        # (4) INFO_LIST_FILE 에 무조건 기록 (누락 방지)
        OUTPUT_DIR = os.path.join(RESULT_DIR, relative_path)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(OUTPUT_DIR, f"output_{pdb_id}.txt")

        with open(INFO_LIST_FILE, 'a') as f:
            f.write(f"{pdb_path}\t{','.join(residues)}\t{output_file}\n")

# ----------------------------
# Write weird cases
# ----------------------------
with open(WEIRD_FILE, 'w') as wf:
    for pdb_id, reason in weird_pdbs.items():
        wf.write(f"{reason}\t{pdb_id}\n")


command = (
    f"{FOLDDISCO} query -i {INDEX_DIR} "
    f"-q {INFO_LIST_FILE} -t 128 -v --sort-by idf --header --format-output tid,node_count,idf,rmsd,tm_score,matching_residues"
)
with open(FOLDDISCO_COMMAND_FILE, 'w') as f:
    f.write(command + "\n")

print("Executing folddisco commands.")
os.system(f"bash {FOLDDISCO_COMMAND_FILE}")
print("Folddisco finished.")

