"""
folddisco_features.py

04(학습) 와 05(평가) 가 반드시 같은 feature 를 계산하도록 한 곳에 모은다.
feature 계산이 두 파일에 복제되면 train/test 불일치가 조용히 생기고, 그건 찾기 매우 어렵다.

FEATURE_NAMES 의 순서가 곧 모델 입력 순서이며, 이 리스트는 .pt 에 함께 저장된다.
"""

import numpy as np
import pandas as pd

ALL_FEATURES = ["z_idf", "z_rmsd", "z_tm", "coverage",
                "idf_pct", "gap_to_top", "log_nhits"]

# 최종: 7개 전부 사용. 5/6/7 비교에서 7개가 scope40 mAP·first-FP 모두 우세했다
# (차이가 시드 노이즈 2σ 를 넘음). --features 로 부분집합 지정 가능.
FEATURE_NAMES = list(ALL_FEATURES)

# 파일마다 컬럼명이 다르다 (AFDB total_combined.txt 는 id/node_count/idf_score/answer_label)
COLUMN_ALIASES = {
    "id": "target_id",
    "node_count": "L",
    "idf_score": "idf",
    "answer_label": "label_raw",
}


def normalize_columns(df):
    df.columns = [c.strip().replace("\r", "") for c in df.columns]
    # 대상 컬럼이 이미 있으면 rename 하지 않는다.
    # (05 는 node_count 를 L 의 복사본으로 만들어두므로, 그대로 rename 하면 L 이 중복되어
    #  df["L"] 이 Series 가 아니라 DataFrame 을 반환한다)
    ren = {k: v for k, v in COLUMN_ALIASES.items()
           if k in df.columns and v not in df.columns}
    df = df.rename(columns=ren)
    dup = df.columns[df.columns.duplicated()].tolist()
    if dup:
        raise SystemExit(f"[FATAL] 중복 컬럼: {dup} — feature 계산이 잘못된다.")
    return df


def _group_stats(qc, s, nq):
    cnt = np.bincount(qc, minlength=nq).astype(np.float64)
    cnt[cnt == 0] = 1.0
    mean = np.bincount(qc, weights=s, minlength=nq) / cnt
    var = np.bincount(qc, weights=s * s, minlength=nq) / cnt - mean ** 2
    return mean, np.sqrt(np.maximum(var, 0.0)) + 1e-9, cnt


def _group_z(qc, s, nq):
    m, sd, _ = _group_stats(qc, s, nq)
    return (s - m[qc]) / sd[qc]


def _group_max(qc, s, nq):
    out = np.full(nq, -np.inf)
    np.maximum.at(out, qc, s)
    return out


def _group_pct(qc, s, nq):
    order = np.lexsort((s, qc))
    q_s = qc[order]
    starts = np.flatnonzero(np.r_[True, q_s[1:] != q_s[:-1]])
    counts = np.diff(np.r_[starts, len(order)])
    within = np.arange(len(order)) - np.repeat(starts, counts)
    pct = np.empty(len(order))
    pct[order] = within / np.maximum(np.repeat(counts, counts) - 1, 1)
    return pct


def build_features(df, query_key, names=None):
    """df 는 normalize_columns 를 거친 상태여야 한다.
    반환: X (float32, [n, len(FEATURE_NAMES)]), qc (group code), nq

    주의: group 통계 기반이므로 df 는 각 query 의 '전체' hit 을 담고 있어야 한다.
    다운샘플된 조각에서 계산하면 학습/평가 분포가 어긋난다.
    """
    qc, uq = pd.factorize(df[query_key].to_numpy(), sort=False)
    qc = qc.astype(np.int64)
    nq = len(uq)

    idf = df["idf"].to_numpy(np.float64)
    rmsd = df["rmsd"].to_numpy(np.float64)
    tm = df["tm_score"].to_numpy(np.float64)
    Lm = df["L"].to_numpy(np.float64)                       # 매칭 크기

    Lq = df["query_num"].astype(str).str.extract(r"length_(\d+)")[0].astype(float).to_numpy()
    Lq = np.where(np.isnan(Lq), Lm, Lq)                     # 파싱 실패 시 fallback

    _, _, cnt = _group_stats(qc, idf, nq)
    gmax = _group_max(qc, idf, nq)

    feats = {
        "z_idf": _group_z(qc, idf, nq),
        "z_rmsd": _group_z(qc, -rmsd, nq),                  # 낮을수록 좋으므로 부호 반전
        "z_tm": _group_z(qc, tm, nq),
        "coverage": Lm / np.maximum(Lq, 1.0),
        "idf_pct": _group_pct(qc, idf, nq),
        "gap_to_top": (gmax[qc] - idf) / (gmax[qc] + 1e-9),
        "log_nhits": np.log1p(cnt[qc]),
    }
    names = list(FEATURE_NAMES if names is None else names)
    unknown = [n for n in names if n not in feats]
    if unknown:
        raise SystemExit(f"[FATAL] 알 수 없는 feature: {unknown}\n        가능: {ALL_FEATURES}")
    X = np.stack([feats[k] for k in names], 1).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), qc, nq


def group_size_guard(qc, nq, name, reference=9228, ratio=10.0):
    """group 당 행 수가 기준보다 크게 작으면 group 통계가 오염된 것이다.
    (M-CSA 26227 / scope40 9228 / AFDB 250K 다운샘플본 175.9)"""
    per = len(qc) / max(nq, 1)
    msg = f"[guard] {name}: {nq} groups, {per:.1f} rows/group (reference {reference})"
    if per * ratio < reference:
        raise SystemExit(
            msg + f"\n[FATAL] group 당 행 수가 기준의 1/{ratio:.0f} 미만이다.\n"
            "        다운샘플된 파일로 보인다. group 통계 기반 feature(z_*, gap_to_top,\n"
            "        log_nhits)가 오염되므로 원본에서 feature 를 먼저 계산한 뒤 행을 줄여야 한다.")
    return msg


def per_query_z(qc, s, nq):
    """query 별 z-score. query 내 monotone 이므로 mAP / first-FP 는 불변,
    pooled 지표(global AP, ROC AUC)만 바뀐다."""
    s = np.asarray(s, dtype=np.float64)
    cnt = np.bincount(qc, minlength=nq).astype(np.float64)
    cnt[cnt == 0] = 1.0
    m = np.bincount(qc, weights=s, minlength=nq) / cnt
    sd = np.sqrt(np.maximum(np.bincount(qc, weights=s * s, minlength=nq) / cnt - m ** 2, 0.0)) + 1e-9
    return (s - m[qc]) / sd[qc]