#!/usr/bin/env bash
# check_mcsa.sh — M-CSA 학습 파일 구조 진단 (한 번의 스트리밍 패스)
# usage: bash check_mcsa.sh <path_to_filtered_combined_MCSA.txt>
set -u
F="${1:?usage: bash check_mcsa.sh <file>}"

echo "=== file: $F ==="
printf "line endings   : "; head -2 "$F" | od -c | grep -q '\\r' && echo "CRLF (\\r 제거하고 처리)" || echo "LF"
echo "--- header ---"
head -1 "$F"
echo

# ---------- [A] 한 패스 통계 ----------
# target_id 는 카디널리티가 커서 여기서 세지 않는다([B]에서 disk sort 로).
LC_ALL=C awk -F'\t' '
NR==1 { sub(/\r$/,"")
        for (i=1;i<=NF;i++) { nm=$i; gsub(/^[ \t]+|[ \t]+$/,"",nm); col[nm]=i }
        need="query_num source target_id L idf rmsd tm_score label_raw"
        n=split(need,w," "); for(i=1;i<=n;i++) if(!(w[i] in col)) miss=miss" "w[i]
        if (miss!="") { printf("[FATAL] missing columns:%s\n", miss) > "/dev/stderr"
                        printf("        -> 줄바꿈(CRLF) 또는 구분자 문제. 아래로 확인:\n") > "/dev/stderr"
                        printf("           file %s ; head -1 %s | od -c | tail -3\n", FN, FN) > "/dev/stderr"
                        fatal=1; exit 2 }
        next }
{ sub(/\r$/,"") }
{
  nf[NF]++
  qn=$col["query_num"]; src=$col["source"]; lab=$col["label_raw"]; idf=$col["idf"]

  lab_cnt[lab]++
  qn_u[qn]++
  src_u[src]++
  key = qn SUBSEP src
  grp_u[key]++
  if (lab+0 != 0) grp_pos[key]++; else grp_neg[key]++

  # 정렬 여부 (스트리밍 청킹 가능한지)
  if (key != prev_key) { if (key in seen_key) unsorted++; seen_key[key]=1; prev_key=key }

  # 비수치/결측 감지
  if (idf !~ /^-?[0-9.eE+-]+$/) bad_idf++
  if ($col["rmsd"] !~ /^-?[0-9.eE+-]+$/) bad_rmsd++
  if ($col["tm_score"] !~ /^-?[0-9.eE+-]+$/) bad_tm++
  total++
}
END {
  if (fatal) exit 2
  printf("[A] rows (excl header)        : %d\n", total)
  printf("[A] field-count histogram     : "); for (k in nf) printf("%d cols x%d  ", k, nf[k]); printf("\n")
  printf("[A] unique query_num          : %d\n", length(qn_u))
  printf("[A] unique source             : %d\n", length(src_u))
  printf("[A] unique (query_num,source) : %d   <-- listwise 학습의 group 수\n", length(grp_u))
  printf("[A] group ordering            : %s\n", (unsorted>0 ? "INTERLEAVED (sort 필요)" : "CONTIGUOUS (스트리밍 가능)"))
  printf("[A] non-numeric idf/rmsd/tm   : %d / %d / %d\n", bad_idf+0, bad_rmsd+0, bad_tm+0)
  printf("[A] label_raw distribution    : ")
  for (l in lab_cnt) printf("%s=%d  ", l, lab_cnt[l]); printf("\n")

  both=0; only_pos=0; only_neg=0; sz=0
  for (k in grp_u) {
    sz += grp_u[k]
    if ((k in grp_pos) && (k in grp_neg)) both++
    else if (k in grp_pos) only_pos++
    else only_neg++
  }
  printf("[A] groups w/ BOTH labels     : %d  <-- 이 값이 0이면 listwise 학습 불가\n", both)
  printf("[A] groups positive-only      : %d\n", only_pos)
  printf("[A] groups negative-only      : %d\n", only_neg)
  printf("[A] mean rows per group       : %.1f\n", sz/length(grp_u))
}' FN="$F" "$F" || exit 2
echo

# ---------- [B] target_id 카디널리티 (disk sort, 메모리 안전) ----------
echo "[B] unique target_id (disk sort, 시간 걸림)..."
TGT=$(LC_ALL=C tail -n +2 "$F" | tr -d '\r' | cut -f3 | LC_ALL=C sort -u -S 2G --parallel=4 | wc -l)
echo "[B] unique target_id          : $TGT"
echo "    -> source 수 << target_id 수 이면 query 는 'source' 이다 (scope40 과 반대)"
echo

# ---------- [C] idf 동점 블록 안에 pos/neg 가 섞여 있는가 ----------
# 새 tie-breaker 학습의 데이터가 여기서 나온다. 파일이 group 별로 연속이라고 가정.
echo "[C] idf-tie 블록 분석 (tie-breaker 학습 신호)"
LC_ALL=C awk -F'\t' '
NR==1 { sub(/\r$/,"")
        for (i=1;i<=NF;i++) { nm=$i; gsub(/^[ \t]+|[ \t]+$/,"",nm); col[nm]=i }
        need="query_num source target_id L idf rmsd tm_score label_raw"
        n=split(need,w," "); for(i=1;i<=n;i++) if(!(w[i] in col)) miss=miss" "w[i]
        if (miss!="") { printf("[FATAL] missing columns:%s\n", miss) > "/dev/stderr"
                        printf("        -> 줄바꿈(CRLF) 또는 구분자 문제. 아래로 확인:\n") > "/dev/stderr"
                        printf("           file %s ; head -1 %s | od -c | tail -3\n", FN, FN) > "/dev/stderr"
                        fatal=1; exit 2 }
        next }
{ sub(/\r$/,"") }
function flush_block(  k,p,n) {
  if (bn=="") return
  n = bpos + bneg
  if (n > 1) { tie_rows += n; tie_blocks++
               if (bpos>0 && bneg>0) { mixed_blocks++; mixed_rows += n
                                       pairs += bpos*bneg } }
  bpos=0; bneg=0
}
{
  key = $col["query_num"] SUBSEP $col["source"] SUBSEP $col["idf"]
  if (key != bn) { flush_block(); bn = key }
  if ($col["label_raw"]+0 != 0) bpos++; else bneg++
  total++
}
END {
  if (fatal) exit 2
  flush_block()
  printf("[C] rows in an idf-tie block  : %d / %d (%.1f%%)\n", tie_rows+0, total, 100*(tie_rows+0)/total)
  printf("[C] tie blocks (size>1)       : %d\n", tie_blocks+0)
  printf("[C] MIXED-label tie blocks    : %d\n", mixed_blocks+0)
  printf("[C] rows in mixed blocks      : %d\n", mixed_rows+0)
  printf("[C] trainable pos-neg pairs   : %d   <-- tie-breaker 학습 표본 수\n", pairs+0)
}' FN="$F" "$F" || exit 2
echo

# ---------- [D] feature 동일 + label 반대 (풀 수 없는 충돌) ----------
echo "[D] feature-identical / label-conflicting 행 (성능 천장)"
LC_ALL=C tail -n +2 "$F" \
 | tr -d '\r' | awk -F'\t' '{print $1"|"$2"|"$4"|"$5"|"$6"|"$7"\t"$8}' \
 | LC_ALL=C sort -S 2G --parallel=4 \
 | LC_ALL=C awk -F'\t' '
     { if ($1==pk) { if ($2!=pl) { conf++; pl="X" } } else { pk=$1; pl=$2 } }
     END { printf("[D] conflicting feature-vectors: %d\n", conf+0) }'
echo "=== done ==="