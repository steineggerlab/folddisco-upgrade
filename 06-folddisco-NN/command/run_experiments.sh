#!/usr/bin/env bash
# run_experiments.sh
#
# 실험 1: 시드 분산  (seed 0/1/2)   -> 지금까지의 차이들이 노이즈인지 판정
# 실험 2: dedupe on/off             -> positive 68.6% 제거가 옳은지 판정
#
# 6개 실행 (3 seed x 2 dedupe). 각각 별도 디렉토리에 저장하므로 덮어쓰기 없음.
# 끝나면 summarize_experiments.py 가 mean±std 표를 만든다.
set -eu

CMD=./command
RESULT_DIR=result_benchmark_v2
AF=$RESULT_DIR/folddisco_results_analyses/afdb_features_cache.npz

# 지금까지 확정된 최적값
FEATS=z_idf,z_rmsd,z_tm,coverage,idf_pct,gap_to_top,log_nhits
BETA=1
SKIP=linear

echo "=============================================="
echo "features : $FEATS"
echo "beta_bce : $BETA   skip: $SKIP   afdb: yes"
echo "=============================================="

for DD in dedupe nodedupe; do
  if [ "$DD" = "dedupe" ]; then DDFLAG="--dedupe"; else DDFLAG=""; fi
  for SEED in 0 1 2; do
    TAG="exp_${DD}_s${SEED}"
    echo ""
    echo "########## $TAG ##########"

    python3 $CMD/04_fit_score.py $DDFLAG \
      --features "$FEATS" --beta_bce $BETA --skip $SKIP --afdb "$AF" \
      --seed $SEED --score_dir "$TAG"

    python3 $CMD/05_test_and_plot_result.py \
      --score_dir "$TAG" --plot_dir "plots_$TAG" --blend_alpha 0.5
  done
done

echo ""
echo "=============================================="
python3 $CMD/summarize_experiments.py --result_dir $RESULT_DIR --prefix exp_