#Python script to test Cα–Cα distance calculation between residues in a PDB file. It reads a specified PDB file, extracts the specified residues, and calculates the distance between their Cα atoms.

from Bio.PDB import PDBParser
import numpy as np
import itertools
import os

PDB_PATH = 'data/classified_pdbs/pdbs_'
RESIDUE_PATH = 'data/chain_residue_list.txt'
	
# ===== 입력 =====
PDB_PATH = 'data/classified_pdbs/pdbs_1_10_101/1eakA01.pdb'
pdbid = '1eakA01'
residue_list = ['A46','A50']  # 테스트용 (원래는 QUERY_INFO에서 읽어올 예정)
# =================

parser = PDBParser(QUIET=True)

def ca_distance(res1, res2):
    """Cα–Cα 거리 계산"""
    if "CA" not in res1 or "CA" not in res2:
        return None
    ca1, ca2 = res1["CA"].coord, res2["CA"].coord
    return np.linalg.norm(ca1 - ca2)

# 파일 체크
if not os.path.exists(PDB_PATH):
    print(f"[WARN] Missing PDB: {PDB_PATH}")
    raise SystemExit

# PDB 파싱
structure = parser.get_structure(pdbid, PDB_PATH)
model = structure[0]   # 첫 번째 모델만 이용

# residue 쌍 조합 생성
pairs = list(itertools.combinations(residue_list, 2))

# 거리 계산
for r1, r2 in pairs:
    chain1, idx1 = r1[0], int(r1[1:])
    chain2, idx2 = r2[0], int(r2[1:])

    try:
        res1 = model[chain1][idx1]
        res2 = model[chain2][idx2]
    except KeyError:
        print(f"[WARN] Missing residue: {pdbid} {r1} or {r2}")
        continue

    dist = ca_distance(res1, res2)
    if dist is not None:
        print(f"{pdbid}\t{r1}-{r2}\t{dist:.3f} Å")
    else:
        print(f"[WARN] No CA atom: {pdbid} {r1} {r2}")
