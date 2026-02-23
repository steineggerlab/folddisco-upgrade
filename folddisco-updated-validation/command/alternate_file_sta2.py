import re
import os
import itertools

path = 'data/sta2/'
input_file = 'folddisco_info_list.txt'
output_file = 'folddisco_info_list_expanded.txt'

input_file = os.path.join(path, input_file)
output_file = os.path.join(path, output_file)

# 2. 조정할 범위 설정 (-3 ~ +3, 0 제외)
offsets = [-7, -6, -5, -4,-3, -2, -1, 1, 2, 3, 4, 5, 6, 7]

change_rules = {
    2: [2],
    3: [2],
    4: [2, 3],
    5: [2, 3, 4],
    6: [2, 3, 4, 5],
    7: [2, 3, 4, 5, 6],
    8: [2, 3, 4, 5, 6, 7],
    9: [2, 3, 4, 5, 6, 7, 8],
    10: [2, 3, 4, 5, 6, 7, 8, 9],
    11: [2, 3, 4, 5, 6, 7, 8, 9, 10],
    12: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    13: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    14: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    15: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    16: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    17: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    18: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    19: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    20: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    21: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    22: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    23: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    24: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    25: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    26: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    27: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
}

# 숫자와 문자를 분리하기 위한 정규표현식 (예: A21 -> A, 21)
pattern = re.compile(r"([A-Za-z]+)(\d+)")

def parse_residue(res_str):
    """A21 같은 문자열을 ('A', 21)로 변환"""
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

        # 원본 잔기 리스트 (예: ['A21', 'A22', 'A25'])
        original_items = parts[1].split(',')
        n = len(original_items)
        
        # 규칙에 해당하는 개수(N)인지 확인
        if n in change_rules:
            target_counts = change_rules[n] # 예: n=4이면 [2, 3]
            
            # 각각의 "바꿀 개수"에 대해 반복 (예: 2개 바꿀 때, 3개 바꿀 때...)
            for k in target_counts:
                
                # 인덱스 조합 생성 (0부터 n-1까지의 인덱스 중 k개를 뽑음)
                # 예: 3개 중 2개를 뽑는다면 -> (0,1), (0,2), (1,2)
                for indices_to_change in itertools.combinations(range(n), k):
                    
                    # 각 오프셋(-3 ~ +3) 적용
                    for offset in offsets:
                        new_items_list = original_items[:] # 원본 복사 (Base)
                        valid_combination = True
                        
                        # 선택된 인덱스들만 값을 변경
                        for idx in indices_to_change:
                            chain, num = parse_residue(original_items[idx])
                            
                            if chain:
                                new_num = num + offset
                                new_items_list[idx] = f"{chain}{new_num}"
                            else:
                                valid_combination = False
                                break
                        
                        if valid_combination:
                            # 결과 저장
                            parts[1] = ",".join(new_items_list)
                            fout.write("\t".join(parts) + "\n")