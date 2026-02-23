import re
import os
import itertools

path = 'data/sta4/'
input_file = 'folddisco_info_list.txt'
output_file = 'folddisco_info_list_expanded.txt'

input_file = os.path.join(path, input_file)
output_file = os.path.join(path, output_file)

# 2. 조정할 범위 설정 (-3 ~ +3, 0 제외)
offsets = [-7, -6, -5, -4,-3, -2, -1, 1, 2, 3, 4, 5, 6, 7]

# 2. 슬라이딩용 오프셋 (고정값 20)
sliding_amount = 20

change_rules = {
    2: [2],
    3: [2],
    4: [2, 3],
    5: [2, 3, 4]
}

# 숫자와 문자를 분리하기 위한 정규표현식 (예: A21 -> A, 21)
pattern = re.compile(r"([A-Za-z]+)(\d+)")

def parse_residue(res_str):
    match = pattern.match(res_str)
    if match:
        return match.group(1), int(match.group(2))
    return None, None

with open(input_file, 'r', encoding='utf-8') as fin, \
     open(output_file, 'w', encoding='utf-8') as fout:

    for line in fin:
        parts = line.strip().split('\t')
        
        if len(parts) < 2:
            continue

        original_items = parts[1].split(',')
        n = len(original_items)
        
        # ==========================================================
        # 작업 1: 기존 조합 변경 로직 (Combinatorial Mutation)
        # ==========================================================
        if n in change_rules:
            target_counts = change_rules[n]
            
            for k in target_counts:
                for indices_to_change in itertools.combinations(range(n), k):
                    for offset in offsets:
                        new_items_list = original_items[:]
                        valid_combination = True
                        
                        for idx in indices_to_change:
                            chain, num = parse_residue(original_items[idx])
                            if chain:
                                new_items_list[idx] = f"{chain}{num + offset}"
                            else:
                                valid_combination = False
                                break
                        
                        if valid_combination:
                            # 조합 변경 결과 저장
                            parts[1] = ",".join(new_items_list)
                            fout.write("\t".join(parts) + "\n")

        # ==========================================================
        # 작업 2: 슬라이딩 로직 (Sliding)
        # 내용: 모든 잔기를 20칸 뒤로(+20) 이동
        # ==========================================================
        sliding_items_list = []
        valid_sliding = True
        
        for item in original_items:
            chain, num = parse_residue(item)
            if chain:
                # 모든 잔기에 대해 +20 적용
                new_num = num + sliding_amount
                sliding_items_list.append(f"{chain}{new_num}")
            else:
                valid_sliding = False
                break
        
        if valid_sliding:
            # 슬라이딩 결과 저장
            # 원본 parts[0](이름 등)는 유지하고 리스트만 교체
            sliding_line = "\t".join([parts[0], ",".join(sliding_items_list)])
            fout.write(sliding_line + "\n")

print("작업 완료")