#Python script to analyze Folddisco results and make score distribution

import numpy as np
import matplotlib.pyplot as plt
import os
import re
import math
from collections import defaultdict
from scipy.stats import gumbel_r

RESULT_DIR = "result/folddisco_result_2_revisited"
PNG_DIR = "result/folddisco_plot"
FILE_INFO = 'combined.txt'

def plot_distribution_score(scores, mu, lam, title="", outfile=None, dpi=300):
    x = np.asarray(scores, float)
    plt.figure(figsize=(6,4))
    plt.hist(x, bins="fd", density=True, alpha=0.5, label="empirical")
    xs = np.linspace(x.min(), x.max(), 400)
    plt.plot(xs, evd_pdf(xs, mu, lam), 'r', label="EVD fit")
    plt.title(f"{title}\nmu={mu:.3f}, lambda={lam:.3f}")
    plt.xlabel("score")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=dpi)  
    plt.close()                     
    print(f"[saved] {outfile}")

def plot_distribution_evalue(evalues, title="", outfile="evd_plot.png", dpi=300):
    x = np.asarray(evalues, dtype=float)
    loc, scale = gumbel_r.fit(x)
    plt.figure(figsize=(6,4))
    plt.hist(x, bins="fd", density=True, alpha=0.5, label="empirical")
    xs = np.linspace(x.min(), x.max(), 400)
    pdf = gumbel_r.pdf(xs, loc=loc, scale=scale)
    plt.plot(xs, pdf, 'r', label=f"EVD fit\nloc={loc:.3f}, scale={scale:.3f}")
    plt.title(title)
    plt.xlabel("e-value")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=dpi)
    plt.close()
    print(f"[saved] {outfile}")

def smart_split(line):
    return line.rstrip("\n").split("\t") if "\t" in line else line.strip().split()

def _lawless416(x, lam):
    ex = np.exp(-lam * x)
    esum = ex.sum()
    xesum = (x*ex).sum()
    xxesum = (x**2*ex).sum()
    f  = 1/lam - x.mean() + xesum/esum
    df = (xesum/esum)**2 - xxesum/esum - 1/(lam*lam)
    return f, df

def _newton_with_bisect(f_df, lam0=0.2, tol=1e-5):
    lam = max(lam0,1e-6)
    for _ in range(100):
        f, df = f_df(lam)
        if abs(f) < tol: return lam
        lam = max(lam - f/df, 1e-6)
    # fallback
    L,R = 1e-6,100
    for _ in range(200):
        M = 0.5*(L+R)
        fM,_ = f_df(M)
        if abs(fM)<tol: return M
        if fM>0: L=M
        else: R=M
    return 0.5*(L+R)

def evd_mle_full(scores):
    x = np.asarray(scores,float)
    lam = _newton_with_bisect(lambda L:_lawless416(x,L))
    mu  = -math.log(np.mean(np.exp(-lam*x)))/lam
    return mu, lam

def evd_sf(x, mu, lam):
    y = lam*(x-mu)
    t = np.exp(-y)
    sf = 1 - np.exp(-t)
    sf[y>20] = np.exp(-y[y>20])   # 큰 y에서 안정화
    return sf

def e_values(scores, mu, lam, M=None):
    p = evd_sf(np.asarray(scores,float), mu, lam)
    if M is None: M = len(scores)
    return M * p

def evd_pdf(x, mu, lam):
    z = lam*(x-mu)
    return lam * np.exp(-z - np.exp(-z))

def pdb_len(pdb_id):
    length = 0
    pdb_id = pdb_id.replace("output_", "")
    pdb_file = f"data/index_pdbs/{pdb_id}.pdb"

    if os.path.exists(pdb_file):
        with open(pdb_file, 'r') as file:
            lines = file.readlines()
            length = len(lines)
    else:
        print(f"PDB file {pdb_file} does not exist.")
    return length

def per_row_output(file_path, out_path):
    # 1) 1-pass: 키별 점수 모으기
    from collections import defaultdict
    buckets = defaultdict(list)
    rows = []  # 행 그대로 저장해두었다가 2-pass에서 e-value 채워서 출력

    with open(file_path, "r") as f:
        header = f.readline().rstrip("\n")
        for line in f:
            cols = smart_split(line)
            if len(cols) < 4:
                continue
            key = cols[1]
            try:
                score = float(cols[4])
            except ValueError:
                continue
            buckets[key].append(score)
            rows.append(cols)

    # 2) 키별 μ,λ 적합
    params = {}
    for key, scores in buckets.items():
        if len(scores) >= 2:
            mu, lam = evd_mle_full(scores)
            M = len(scores)    
            params[key] = (mu, lam, len(scores))  # (μ, λ, M)

    # 4) 각 행에 e-value 계산해서 "행 단위"로 모두 출력
    with open(out_path, "w") as out:
        out.write("source\tid\tnode_count\tidf_score\te_value\n")
        for cols in rows:
            if len(cols) < 4:
                continue
            key = cols[1]
            try:
                score = float(cols[4])
            except ValueError:
                continue

            if key in params:
                mu, lam, M = params[key]
                e = e_values([score], mu, lam, M=M)[0]
                out.write(f"{cols[0]}\t{cols[1]}\t{cols[2]}\t{cols[3]}\t{cols[4]}\t{e:.3e}\n")

            else: # 표본 1개 등으로 파라미터 없으면 
                out.write(f"{cols[0]}\t{cols[1]}\t{cols[2]}\t{cols[3]}\t{cols[4]}\tNA\n")

def main():
    print("Starting Folddisco result to distribution analysis...")

    for file in os.listdir(RESULT_DIR):
        if FILE_INFO not in file:
            continue
        file_path   = os.path.join(RESULT_DIR, file)
        RESULT_FILE = os.path.join(RESULT_DIR, f"{file[0]}_evalues.txt")
        print(f"[INFO] Processing {file_path} ...")
        per_row_output(file_path, RESULT_FILE)
        print(f"[OK] Saved per-row e-values -> {RESULT_FILE}")
    

if __name__ == "__main__":
    main()

