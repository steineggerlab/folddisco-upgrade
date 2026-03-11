#Python script to generate alternative residue lists based on original data, applying symmetry and random transformations while ensuring structural validity against PDB files.

import re
import os
import random  # [수정] Random as random -> import random (표준 라이브러리)
from Bio.PDB import PDBParser

path = 'data/'
input_file_name = 'folddisco_info_list.txt'
output_file_name = 'folddisco_info_list_expanded.txt'
parser = PDBParser(QUIET=True)
random.seed(42)  # 재현성을 위한 시드 설정

input_file = os.path.join(path, input_file_name)
output_file = os.path.join(path, output_file_name)

pattern = re.compile(r"([A-Za-z]+)(\d+)")

def parse_residue(res_str):
    match = pattern.match(res_str)
    if match:
        return match.group(1), int(match.group(2))
    return None, None

def format_residues(items):
    items.sort(key=lambda x: x['num'])
    return ",".join([f"{item['chain']}{item['num']}" for item in items])

# [추가] 해당 체인의 유효한 잔기 번호 리스트를 추출하는 함수
def get_valid_residue_numbers(structure, target_chain_id):
    valid_nums = []
    for model in structure:
        for chain in model:
            if chain.id == target_chain_id:
                for r in chain.get_residues():
                    # HETATM(물 분자 등) 제외하고 표준 잔기만
                    if r.id[0] == ' ':
                        valid_nums.append(r.id[1])
                return valid_nums # 첫 번째 모델의 해당 체인만 찾으면 반환
    return []

# --- 메인 로직 ---
with open(input_file, 'r', encoding='utf-8') as fin, \
     open(output_file, 'w', encoding='utf-8') as fout:

    output_line_idx = 0
    
    # [최적화] 이전 루프의 PDB ID를 기억하여 중복 로딩 방지 (간단한 캐싱)
    prev_pdb_path = ""
    current_structure = None

    for line in fin:
        parts = line.strip().split('\t')
        
        if len(parts) < 3:
            continue
            
        pdb_file_path = parts[0].replace('..', '.')
        base_file_path = parts[2]

        # ---------------------------------------------------------
        # 1. 수정된 원본 라인 기록
        # ---------------------------------------------------------
        #parts[2] = f"{base_file_path}{output_line_idx}.txt"
        #fout.write("\t".join(parts) + "\n")
        #output_line_idx += 1

        # ---------------------------------------------------------
        # 2. 데이터 파싱
        # ---------------------------------------------------------
        original_str_items = parts[1].split(',')
        residues = []
        is_valid_line = True
        
        for item in original_str_items:
            chain, num = parse_residue(item)
            if chain is None:
                is_valid_line = False
                break
            residues.append({'chain': chain, 'num': num})
        
        if not is_valid_line:
            continue
            
        n = len(residues)

        # ==========================================
        # Logic 1: 대칭 이동 (Symmetry Reflection)
        # ==========================================
        for i in range(n):
            pivot = residues[i]
            pivot_num = pivot['num']
            new_items_list = []
            
            for j in range(n):
                target = residues[j]
                if i == j:
                    new_items_list.append(target)
                else:
                    diff = target['num'] - pivot_num
                    new_num = pivot_num - diff
                    # 주의: 대칭 이동으로 나온 new_num도 실제 PDB에 없을 수 있음.
                    # 단순 기하학적 대칭이라면 유지하되, 구조적 유효성이 중요하다면 체크 필요.
                    if new_num > 0: 
                        new_items_list.append({'chain': target['chain'], 'num': new_num})
            
            output_parts = parts[:]
            output_parts[1] = format_residues(new_items_list)
            output_parts[2] = f"{base_file_path}{output_line_idx}.txt"
            fout.write("\t".join(output_parts) + "\n")
            output_line_idx += 1

        # ==========================================
        # Logic 2: 랜덤 변환 (Random Transformations) - [대폭 수정됨]
        # ==========================================       
        if pdb_file_path != prev_pdb_path:
            try:
                pdb_id = os.path.splitext(os.path.basename(pdb_file_path))[0]
                current_structure = parser.get_structure(pdb_id, pdb_file_path)
                prev_pdb_path = pdb_file_path
            except Exception as e:
                print(f"Error parsing PDB {pdb_file_path}: {e}")
                continue

        if current_structure is None:
            continue

        for pivot_idx in range(n):
            pivot_residue = residues[pivot_idx] # 현재 루프의 주인공(고정될 피벗)
            pivot_num = pivot_residue['num']

            chain_pools = {}
            for residue in residues:
                # 피벗은 그대로 유지
                if residue == pivot_residue:
                    new_items_list.append(residue)
                    continue
                
                else:
                    # 나머지 잔기들은 랜덤 변환
                    target_chain = residue['chain']
                    current_num = residue['num']
                    
                    # 해당 체인에 실제로 존재하는 잔기 번호들 가져오기
                    valid_candidates = get_valid_residue_numbers(current_structure, target_chain)
                    
                    # 자기 자신 제외
                    valid_candidates = [x for x in valid_candidates if x != current_num]

                    if not valid_candidates:
                        # 후보가 없으면 원래 값 유지
                        residue_num = current_num
                    else:
                        # 실제 존재하는 번호 중에서 랜덤 선택
                        residue_num = pivot_num + identifier
                        identifier += 1
                        if residue_num not in valid_candidates:
                            residue_num = random.choice(valid_candidates)
                    new_items_list.append({'chain': target_chain, 'num': residue_num})

            # [출력 작성] 피벗 하나에 대한 처리가 끝날 때마다 파일에 기록
            output_parts = parts[:]
            output_parts[1] = format_residues(new_items_list)
            
            # 고유 번호 부여
            output_parts[2] = f"{base_file_path}{output_line_idx}.txt"
            
            fout.write("\t".join(output_parts) + "\n")
            output_line_idx += 1
print(f"완료: {output_file} 생성됨. 총 {output_line_idx}개의 라인이 작성되었습니다.")