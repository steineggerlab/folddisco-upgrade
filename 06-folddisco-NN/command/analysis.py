import os
import pandas as pd

# CSV 파일들이 위치한 디렉토리 경로
results_dir = "result_benchmark_v2/folddisco_results_scores_final"

# 1. Folddisco-NN에서 True Positive로 잡힌 사례들 읽기
nn_tp_path = os.path.join(results_dir, "Folddisco-NN_true_positives_top.csv")
df_nn_tp = pd.read_csv(nn_tp_path)

# 비교 대상이 되는 다른 모델/베이스라인들의 False Negative CSV 파일 목록
other_fn_files = [
    "Baseline_IDF_false_negatives_top.csv",
    "Baseline_RMSD_false_negatives_top.csv",
    "Baseline_RMSD_Node_false_negatives_top.csv",
    "Baseline_TM_false_negatives_top.csv",
    "IDF_then_RMSD_false_negatives_top.csv"
]

# Pair 또는 Query/Target 식별을 위한 키 컬럼 설정
# CSV에 있는 컬럼명에 맞춰 수정해 주세요 (예: 'query', 'target' 또는 'pair_id', 'query_id' 등)
# 보통 query, target 두 컬럼으로 쌍을 식별합니다.
key_cols = ["source", "target_id"]  # 만약 'pair_id' 단일 컬럼이라면 ['pair_id']로 변경

# 2. 다른 베이스라인들에서 모두 False Negative로 잡힌 페어들의 교집합 구하기
common_fn_pairs = None

for fn_file in other_fn_files:
    file_path = os.path.join(results_dir, fn_file)
    if os.path.exists(file_path):
        df_fn = pd.read_csv(file_path)
        
        # 식별 키 추출 (튜플 형태의 set으로 변환)
        fn_pairs = set(zip(*[df_fn[col] for col in key_cols]))
        
        if common_fn_pairs is None:
            common_fn_pairs = fn_pairs
        else:
            # 다른 모델에서도 '모두' FN이었던 건만 교집합(intersection)
            common_fn_pairs = common_fn_pairs.intersection(fn_pairs)

print(f"다른 모델에서 모두 FN이었던 페어의 수: {len(common_fn_pairs)}")

# 3. Folddisco-NN의 TP 데이터 중, 다른 모델에서는 모두 FN이었던 사례만 필터링
def is_in_common_fn(row):
    return tuple(row[col] for col in key_cols) in common_fn_pairs

unique_tp_df = df_nn_tp[df_nn_tp.apply(is_in_common_fn, axis=1)]

# 4. 상위 예시 추출 및 CSV로 저장
output_path = os.path.join(results_dir, "Folddisco-NN_unique_true_positives_examples.csv")
unique_tp_df.to_csv(output_path, index=False)

print(f"총 {len(unique_tp_df)}개의 Folddisco-NN 단독 성공 사례를 찾았습니다.")
print(f"결과가 다음 경로에 저장되었습니다: {output_path}")