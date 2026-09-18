"""
summarize_experiments.py

run_experiments.sh 가 만든 로그들을 읽어 mean±std 표를 만든다.

판정 기준:
  - 시드 표준편차가 지금까지 비교해온 차이(0.004~0.02)보다 크면,
    그 비교들은 노이즈였다는 뜻이다.
  - dedupe on/off 는 시드 분산과 견주어 판단한다 (차이 > 2*std 여야 실재).
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--result_dir", default="result_benchmark_v2")
p.add_argument("--prefix", default="exp_")
p.add_argument("--out", default="")
args = p.parse_args()

MODELS = ["NN_Model", "NN_Model_z", "Blend_a0.50", "Baseline_IDF", "IDF_then_RMSD"]


def parse_log(path):
    """05 의 analysis_log.txt 에서 지표를 뽑는다."""
    txt = open(path, encoding="utf-8", errors="ignore").read()
    out, cur = {}, None
    for line in txt.splitlines():
        m = re.match(r"--- Evaluating Model: (\S+) ---", line)
        if m:
            cur = m.group(1)
            out.setdefault(cur, {})
            continue
        if cur:
            for key, pat in (("globalAP", r"Global Average Precision \(Pooled\):\s*([\d.]+)"),
                             ("mAP", r"Mean Average Precision \(mAP\):\s*([\d.]+)"),
                             ("rocauc", r"ROC AUC:\s*([\d.]+)"),
                             ("f1", r"F1 Score:\s*([\d.]+)")):
                mm = re.search(pat, line)
                if mm:
                    out[cur][key] = float(mm.group(1))
        mm = re.search(r"\[first-FP/pessimistic\] (\S+): AUC=([\d.]+)", line)
        if mm:
            out.setdefault(mm.group(1), {})["firstFP"] = float(mm.group(2))
    return out


rows = []
for d in sorted(glob.glob(os.path.join(args.result_dir, args.prefix + "*"))):
    log = os.path.join(d, "analysis_log.txt")
    if not os.path.exists(log):
        print(f"[skip] {d} (analysis_log.txt 없음)")
        continue
    tag = os.path.basename(d)
    m = re.match(rf"{re.escape(args.prefix)}(\w+?)_s(\d+)", tag)
    setting, seed = (m.group(1), int(m.group(2))) if m else (tag, -1)
    for model, met in parse_log(log).items():
        if model in MODELS and met:
            rows.append(dict(setting=setting, seed=seed, model=model, **met))

if not rows:
    raise SystemExit("[FATAL] 파싱된 결과가 없다. run_experiments.sh 가 끝났는지 확인할 것.")

df = pd.DataFrame(rows)
METRICS = [c for c in ("globalAP", "mAP", "firstFP", "rocauc", "f1") if c in df.columns]

print("\n" + "=" * 86)
print("실험 1+2 요약  (seed 별 mean ± std)")
print("=" * 86)
print(f"{'setting':<12}{'model':<16}" + "".join(f"{m:>15}" for m in METRICS))
print("-" * 86)
for (st, mo), g in df.groupby(["setting", "model"], sort=False):
    cells = "".join(f"{g[m].mean():>9.4f}±{g[m].std(ddof=1) if len(g) > 1 else 0:.4f}"
                    for m in METRICS)
    print(f"{st:<12}{mo:<16}{cells}")

# ---------- 실험 1: 시드 노이즈 ----------
print("\n" + "=" * 86)
print("실험 1 — 시드 노이즈 크기")
print("=" * 86)
nn = df[df.model == "NN_Model_z"]
for st, g in nn.groupby("setting"):
    if len(g) < 2:
        continue
    print(f"[{st}] n={len(g)}")
    for m in METRICS:
        sd = g[m].std(ddof=1)
        rng = g[m].max() - g[m].min()
        verdict = ("노이즈 큼 — 0.02 이하 차이는 무의미" if sd > 0.01 else
                   "보통 — 0.01 이하 차이는 주의" if sd > 0.005 else
                   "작음 — 기존 비교 유효")
        print(f"   {m:<10} std={sd:.4f}  range={rng:.4f}   {verdict}")

# ---------- 실험 2: dedupe ----------
print("\n" + "=" * 86)
print("실험 2 — dedupe on/off (시드 분산 대비)")
print("=" * 86)
if nn.setting.nunique() >= 2:
    a = nn[nn.setting == "dedupe"]
    b = nn[nn.setting == "nodedupe"]
    if len(a) and len(b):
        for m in METRICS:
            d = b[m].mean() - a[m].mean()
            pooled = np.sqrt((a[m].var(ddof=1) + b[m].var(ddof=1)) / 2) if min(len(a), len(b)) > 1 else 0
            sig = "실재" if pooled > 0 and abs(d) > 2 * pooled else "노이즈 구간"
            print(f"   {m:<10} nodedupe−dedupe = {d:+.4f}   pooled_std={pooled:.4f}   -> {sig}")
        print("\n   양수면 dedupe 를 끄는 쪽이 낫다 (positive 3배 유지).")
        print("   '노이즈 구간' 이면 단순한 쪽(dedupe 유지)을 택하면 된다.")
else:
    print("   dedupe/nodedupe 양쪽 결과가 모두 필요하다.")

out = args.out or os.path.join(args.result_dir, "experiments_summary.csv")
df.to_csv(out, index=False)
print(f"\n[save] {out}")
print("=" * 86)