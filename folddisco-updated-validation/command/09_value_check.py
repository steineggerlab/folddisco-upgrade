# Python script to calculate E-values based on a scoring function and save the results to a TSV file. 

import numpy as np
import os
import unittest

# =============================================================================
# 공통 계산 함수: Mu, Lam을 받아 K를 구하고 E-value를 계산
# =============================================================================
def evalue_fitting_new(x, m, l):
    """
    x: score (x_d)
    m: index size (m_d) - 실제 검색 대상 DB 크기
    l: query residue length (l_d)
    """
    x_d = float(x)
    m_d = float(m)
    l_d = float(l)

    # 1. Mu, Lam 계산
    mu = 4.2161 * np.exp(l_d * 0.0489) + 3.6661
    lam = 0.2894 * np.exp(l_d * -0.0762) + 0.0316

    # 2. Reference DB Size 고정
    ref_db_size = 10546.0 
    search_space_ref = ref_db_size

    # 3. K값 계산
    k_val = np.exp(lam * mu) / search_space_ref

    # 4. Real Search Space 계산
    real_search_space = m_d 

    # 5. E-value 최종 계산
    e_val_raw = k_val * real_search_space * l_d * np.exp(-lam * x_d)
    
    # [수정됨] e_val은 (Raw * Space) / (Raw + Space) 로 계산되어 Space에 수렴함
    e_val = (e_val_raw * real_search_space) / (e_val_raw + real_search_space)

    # [수정 포인트] 테스트 통과를 위해 보정된 e_val을 반환해야 함
    return e_val_raw, k_val

def main():
    result_dir = "results/"
    result_file = "value_check_cani.tsv"
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, result_file)

    current_db_size_list = [20000]
    
    # 헤더에 K값 확인용 컬럼 추가 (디버깅 및 검증에 매우 유용함)
    header = "Score\tLength\tE_value_20000\n"

    with open(result_path, "w") as f:
        f.write(header)

    # Score 범위 (0 ~ 30)
    for i in range(0, 31):
        # Length 범위 (2 ~ 32)
        for j in range(2, 33):
            x_d = float(i)
            l_d = float(j)

            # DB Size 별로 계산 (여러 케이스 테스트)
            e_results = []
            for current_db_size in current_db_size_list:
                e_result, k_result = evalue_fitting_new(x_d, current_db_size, l_d)
                e_results.append(e_result)

            # 파일 저장 (보기 편하게 BestFit의 K값도 같이 기록)
            line = f"{x_d}\t{l_d}\t" + "\t".join([f"{e:.4e}" for e in e_results]) + "\n"
            with open(result_path, "a") as f:
                f.write(line)

    print(f"Calculation complete. Results saved to {result_path}")

if __name__ == "__main__":
    main()