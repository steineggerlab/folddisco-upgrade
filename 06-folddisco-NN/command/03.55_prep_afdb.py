"""
03_prep_afdb.py

total_combined.txt (18GB, AFDB shuffle negative 원본) 에서
  1) group(=source) 단위로 스트리밍하며
  2) group '전체' 로 feature 를 계산한 뒤
  3) 행을 다운샘플
해서 캐시(.npz)로 떨군다.

순서가 핵심이다. 기존 data_negative_AFDB_shuffle.txt 는 먼저 다운샘플한 결과라
group 당 175.9행 밖에 없고(원본은 ~176,000행), z_idf / idf_pct / gap_to_top / log_nhits
같은 group 통계 기반 feature 가 전부 오염되어 있다. 그 파일은 새 feature set 에 쓸 수 없다.

컬럼명이 M-CSA 와 다르다 (id / node_count / idf_score / answer_label)
-> folddisco_features.COLUMN_ALIASES 가 처리한다.

usage:
  python 03_prep_afdb.py --keep_per_group 300
  python 03_prep_afdb.py --keep_per_group 300 --max_groups 50   # 빠른 확인용
"""

import os
import time
import argparse
import numpy as np
import pandas as pd

from folddisco_features import (FEATURE_NAMES, ALL_FEATURES,
                                normalize_columns, build_features)

RESULT_DIR = "result_benchmark_v2"
ANALYSES = os.path.join(RESULT_DIR, "folddisco_results_analyses")
SRC = os.path.join(ANALYSES, "total_combined.txt")
OUT = os.path.join(ANALYSES, "afdb_features_cache.npz")

p = argparse.ArgumentParser()
p.add_argument("--src", default=SRC)
p.add_argument("--out", default=OUT)
p.add_argument("--query_key", default="source")
p.add_argument("--keep_per_group", type=int, default=300,
               help="group 당 유지할 행 수 (feature 계산 '후' 다운샘플)")
p.add_argument("--chunksize", type=int, default=2_000_000)
p.add_argument("--max_groups", type=int, default=0, help="0=전체")
p.add_argument("--min_group_rows", type=int, default=1000,
               help="이보다 작은 group 은 통계가 불안정하므로 버린다")
p.add_argument("--features", default=",".join(ALL_FEATURES),
               help="캐시에 저장할 feature. 기본은 전체(상위집합)라 부분집합은 재생성 불필요. "
                    f"가능: {','.join(ALL_FEATURES)}")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

rng = np.random.default_rng(args.seed)
FEATS = [f.strip() for f in args.features.split(",") if f.strip()]
USECOLS = ["query_num", "source", "id", "node_count", "idf_score", "rmsd",
           "tm_score", "answer_label"]

print(f"[src] {args.src}")
print(f"[cfg] keep_per_group={args.keep_per_group}  min_group_rows={args.min_group_rows}")
print(f"[cfg] features={FEATS}")
if set(FEATS) >= set(ALL_FEATURES):
    print("[cfg] 전체 feature 저장 -> 04 에서 어떤 부분집합을 골라도 재생성 불필요")

feat_chunks, lab_chunks, grp_chunks = [], [], []
n_groups = n_rows_seen = n_rows_kept = n_dropped_small = 0
t0 = time.time()


def flush(buf):
    """한 group 전체를 받아 feature 를 계산하고 다운샘플한다."""
    global n_groups, n_rows_seen, n_rows_kept, n_dropped_small
    g = pd.concat(buf, ignore_index=True) if len(buf) > 1 else buf[0]
    n_rows_seen += len(g)
    if len(g) < args.min_group_rows:
        n_dropped_small += 1
        return
    g = normalize_columns(g)
    g = g[g["idf"].notna() & g["tm_score"].notna() & (g["L"] > 1) & (g["L"] <= 32)]
    if len(g) < args.min_group_rows:
        n_dropped_small += 1
        return
    g["rmsd"] = g["rmsd"].fillna(999.0)
    g = g.reset_index(drop=True)

    # feature 는 group 전체로 계산한다 (다운샘플 전)
    X, _, _ = build_features(g, args.query_key, FEATS)
    y = np.where(g["label_raw"].to_numpy() == 0, 0, 1).astype(np.int8)

    # 그 다음 행을 줄인다
    k = min(args.keep_per_group, len(X))
    sel = rng.choice(len(X), k, replace=False)
    feat_chunks.append(X[sel])
    lab_chunks.append(y[sel])
    grp_chunks.append(np.full(k, n_groups, dtype=np.int32))
    n_groups += 1
    n_rows_kept += k
    if n_groups % 50 == 0:
        print(f"  groups={n_groups}  seen={n_rows_seen:,}  kept={n_rows_kept:,}  "
              f"({time.time()-t0:.0f}s)", flush=True)


buf, cur = [], None
stop = False
reader = pd.read_csv(args.src, sep="\t", engine="c", chunksize=args.chunksize,
                     usecols=lambda c: c.strip().replace("\r", "") in USECOLS,
                     dtype={"query_num": str, "source": str, "id": str})

for ch in reader:
    ch.columns = [c.strip().replace("\r", "") for c in ch.columns]
    for c in ("query_num", "source", "id"):
        if c in ch.columns:
            ch[c] = ch[c].astype(str).str.replace("\r", "", regex=False)
    # group 이 파일 내에서 연속(CONTIGUOUS)이라는 진단 결과에 의존한다
    key = ch["query_num"] + "|" + ch["source"]
    for k, sub in ch.groupby(key, sort=False):
        if cur is None or k == cur:
            buf.append(sub)
            cur = k
        else:
            flush(buf)
            buf, cur = [sub], k
            if args.max_groups and n_groups >= args.max_groups:
                stop = True
                break
    if stop:
        break

if buf and not stop:
    flush(buf)

if n_groups == 0:
    raise SystemExit("[FATAL] 유효한 group 이 없다. --min_group_rows 를 낮추거나 파일을 확인할 것.")

X = np.concatenate(feat_chunks).astype(np.float32)
y = np.concatenate(lab_chunks)
gid = np.concatenate(grp_chunks)

np.savez_compressed(args.out, X=X, y=y, gid=gid,
                    feature_names=np.array(FEATS, dtype=object),
                    rows_seen=n_rows_seen, keep_per_group=args.keep_per_group)

print("\n" + "=" * 70)
print(f"[done] groups={n_groups}  (너무 작아 버린 group={n_dropped_small})")
print(f"[done] 원본에서 본 행={n_rows_seen:,}  -> 캐시 행={n_rows_kept:,} "
      f"({100*n_rows_kept/max(n_rows_seen,1):.2f}%)")
print(f"[done] group 당 평균 원본 행={n_rows_seen/max(n_groups,1):,.0f} "
      f"(feature 는 이 전체로 계산됨)")
print(f"[done] positives={int(y.sum())} / {len(y)}")
print(f"[save] {args.out}  features={FEATS}")
print("=" * 70)