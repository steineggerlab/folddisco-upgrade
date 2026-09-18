#!/bin/bash
set -e

echo "Starting pipeline"

# 3. 모든 파이썬 스크립트에 변환된 인자를 전달하여 실행
#python3 -u ./command/01_folddisco_validation_query.py
#python3 -u ./command/01.5_folddisco_result_summarize.py
#python3 ./command/02_classify_folddisco_result_by_querylen.py
#python3 ./command/03_folddisco_gather_by_querylen.py
#python3 ./command/03.5_filter_data.py
AF=result_benchmark_v2/folddisco_results_analyses/afdb_features_cache.npz
SCORE_DIR=folddisco_results_scores_final2
PLOT_DIR=analyses_plot_comparison_final2
 
# feature 7개 / beta_bce 1 / skip linear 은 04 의 기본값. dedupe 는 쓰지 않는다.
python3 ./command/04_fit_score.py --afdb "$AF" --seed 0 --score_dir "$SCORE_DIR"
 
# main figure: Baseline_IDF / IDF_then_RMSD / Folddisco-NN
python3 ./command/05_test_and_plot_result.py \
  --score_dir "$SCORE_DIR" --plot_dir "$PLOT_DIR" --main_figure
 
# supplementary: 전체 baseline + ablation 변형(NN_raw_logit, Blend)
#python3 ./command/05_test_and_plot_result.py \
#  --score_dir "$SCORE_DIR" --plot_dir "${PLOT_DIR}_all" --all_variants

#python3 ./command/analysis.py
 
echo "Final run finished successfully."