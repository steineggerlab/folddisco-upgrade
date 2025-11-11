#Python script to convert Folddisco validation results into plot-ready format

import os, math, random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import kstest

DATA = "result/folddisco_results_analyses/total_evalues_fitted_rmsd.txt"
OUTD = "result/folddisco_results_analyses/qc_rmsd_evalue"
os.makedirs(OUTD, exist_ok=True)

# ----- 설정 -----
TS = [0.1, 0.5, 1, 2, 5, 10]    # E 임계값들
SAMPLE_N = 200000               # 큰 길이 bin은 플롯용으로 표본 추출

def parse_len(s):  # "9_combined.txt" -> 9
    try: return int(s.split("_")[0])
    except: return 1

def is_self(query, index):
    q = query.replace("output_","").replace(".txt","")
    i = index.replace("data/index_pdbs/","").replace(".pdb","")
    return q == i

# ----- 1pass: non-self E를 길이별로 QC 집계 -----
# 메모리 폭발 방지: 플롯/KS용으로는 reservoir sample만 보관
counts = {L: { 'N':0, 'under':{t:0 for t in TS} } for L in range(2, 100)}
samples = defaultdict(list)  # L -> sampled E (non-self)

with open(DATA, "r") as f:
    hdr = f.readline().strip().split("\t")
    col = {name:i for i,name in enumerate(hdr)}
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 6: continue
        L   = parse_len(parts[col["query_num"]])
        q   = parts[col["source"]]
        idx = parts[col["id"]]
        try:
            E = float(parts[col["e_value"]])
        except:
            continue
        if is_self(q, idx):
            continue  # non-self만 QC

        # 집계
        counts.setdefault(L, { 'N':0, 'under':{t:0 for t in TS} })
        counts[L]['N'] += 1
        for t in TS:
            if E <= t:
                counts[L]['under'][t] += 1

        # reservoir sampling
        arr = samples[L]
        n = len(arr)
        if n < SAMPLE_N:
            arr.append(E)
        else:
            j = random.randint(0, counts[L]['N']-1)
            if j < SAMPLE_N:
                arr[j] = E

# ----- 길이별 QC 테이블 작성 -----
qc_lines = []
qc_lines.append("L\tN_nonself\t" + "\t".join([f"obs_frac@{t}" for t in TS]) + "\t" +
                "\t".join([f"theory@{t}" for t in TS]) + "\t" +
                "\t".join([f"ratio(obs/theory)@{t}" for t in TS]))
for L in sorted(k for k in counts if counts[k]['N']>0):
    N = counts[L]['N']
    obs = [counts[L]['under'][t]/N for t in TS]
    th  = [1.0 - math.exp(-t) for t in TS]
    rat = [(o/max(tt,1e-12)) for o,tt in zip(obs,th)]
    qc_lines.append(
        str(L)+"\t"+str(N)+"\t"+
        "\t".join(f"{x:.4f}" for x in obs)+"\t"+
        "\t".join(f"{x:.4f}" for x in th)+"\t"+
        "\t".join(f"{x:.3f}" for x in rat)
    )

qc_path = os.path.join(OUTD, "qc_by_length_rmsd.tsv")
with open(qc_path, "w") as w: w.write("\n".join(qc_lines)+"\n")
print(f"[OK] saved {qc_path}")

# ----- 플롯: ECDF vs theory & survival (semilog), 길이별 샘플 몇 개 -----
def ecdf(a):
    a = np.sort(np.asarray(a, float))
    y = np.arange(1, len(a)+1)/len(a)
    return a, y

picked = [L for L in sorted(samples.keys()) if len(samples[L])>=1000]
picked = picked[:6]  # 너무 많으면 상위 6개만
for L in picked:
    e = np.asarray(samples[L], float)
    e = e[np.isfinite(e)]
    if len(e) < 100: continue

    # KS test vs Exp(1): scipy expects CDF name 'expon' (scale=1, loc=0)
    D, p = kstest(e, 'expon')
    # ECDF vs theory
    x,y = ecdf(e)
    t  = np.linspace(0, np.percentile(e, 99), 200)
    F  = 1.0 - np.exp(-t)

    plt.figure(figsize=(6,4))
    plt.plot(x, y, label=f"ECDF non-self (L={L})")
    plt.plot(t, F, label="theory Exp(1)")
    plt.xlabel("E-value"); plt.ylabel("CDF")
    plt.title(f"ECDF vs theory (KS D={D:.3f}, p={p:.1e})")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTD, f"ecdf_L{L}_rmsd.png"), dpi=200)
    plt.close()

    # Survival on semilog: straight line with slope -1 if Exp(1)
    from numpy import maximum
    T = np.linspace(0, np.percentile(e, 99), 200)
    S_th = np.exp(-T)
    # empirical survival
    x_sorted = np.sort(e)
    surv = 1.0 - np.arange(1, len(x_sorted)+1)/len(x_sorted)

    plt.figure(figsize=(6,4))
    plt.semilogy(x_sorted, np.maximum(surv, 1e-12), label=f"empirical S (L={L})")
    plt.semilogy(T, S_th, label="theory S=exp(-t)")
    plt.xlabel("E-value"); plt.ylabel("Survival (log scale)")
    plt.title("Survival vs Exp(1) (semilog)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTD, f"survival_L{L}_rmsd.png"), dpi=200)
    plt.close()

print(f"[OK] saved plots into {OUTD}")