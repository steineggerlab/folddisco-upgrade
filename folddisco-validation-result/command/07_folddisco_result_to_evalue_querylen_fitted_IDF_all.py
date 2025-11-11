#Python script to analyze Folddisco results and make score distribution

import numpy as np
import matplotlib.pyplot as plt
import os
import re
import math
from collections import defaultdict
from scipy.stats import gumbel_r
from scipy.stats import gaussian_kde
from multiprocessing import Pool

RESULT_DIR = "result/folddisco_result_2_revisited"
PNG_DIR = "result/folddisco_plot"
FILE_INFO = 'total_combined.txt'
INDEX_DIR = "data/index_pdbs"
N_INDEX = sum(1 for f in os.listdir(INDEX_DIR) if f.endswith(".pdb"))
M_EFF = max(N_INDEX - 1, 1)

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
    pdb_id = pdb_id.replace("output_", "").replace(".txt", "")
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
    query_lens = {}
    rows = []  # 행 그대로 저장해두었다가 2-pass에서 e-value 채워서 출력

    dir_name = os.path.basename(os.path.dirname(file_path))
    match = re.search(r'(\d+)', dir_name)
    if match:
        L_dir = int(match.group(1))
    else:
        print(f"[WARN] Cannot parse length from {dir_name}, default=1")
        L_dir = 1


    with open(file_path, "r") as f:
        header = f.readline().rstrip("\n")
        for line in f:
            cols = smart_split(line)
            if len(cols) < 5:
                continue
            key = cols[1]
            try:
                score = float(cols[4])
            except ValueError:
                continue
            buckets[key].append(score)
            rows.append(cols)
            query_lens[key] = L_dir

    # 2) 키별 μ,λ 적합
    params = {}
    for key, scores in buckets.items():
        if len(scores) >= 2:
            mu, lam = evd_mle_full(scores)
            L = query_lens.get(key, 1)
            params[key] = (mu, lam, M_EFF, L)  # (μ, λ, M)

    
    # 3) μ, λ와 log(L_q)의 관계 회귀
    if params:
        lengths = np.array([v[3] for v in params.values()], float)
        mus     = np.array([v[0] for v in params.values()], float)
        lams    = np.array([v[1] for v in params.values()], float)
        logL = np.log(lengths + 1e-8)

        # log-space 회귀: lam(L) = exp(a + b*logL)
        # mu(L) = c + d*logL
        a,b = np.polyfit(logL, np.log(lams+1e-8), 1)
        c,d = np.polyfit(logL, mus, 1)
    else:
        a=b=c=d=0.0

    def predict_params(L):
        lam_pred = np.exp(a + b*np.log(max(L,1)))
        mu_pred  = c + d*np.log(max(L,1))
        return mu_pred, lam_pred
    
    p_self, p_nonself = [], []
    # 4) 각 행에 e-value 계산해서 "행 단위"로 모두 출력
    with open(out_path, "w") as out:
        out.write("query_num\tsource\tid\tnode_count\tIDF_score\te_value\n")
        for cols in rows:
            if len(cols) < 5:
                continue
            key = cols[1]
            target = cols[2]
            try:
                score = float(cols[4])
            except ValueError:
                continue

            L = query_lens.get(key, 1)

            mu_pred, lam_pred = predict_params(L)
            p = evd_sf([score], mu_pred, lam_pred)[0]
            e = len(buckets[key]) * p
            out.write(f"{cols[0]}\t{cols[1]}\t{cols[2]}\t{cols[3]}\t{cols[4]}\t{e:.3e}\n")

            # self / non-self 분류
            q_short = key.replace("output_", "").replace(".txt", "")
            idx_short = target.replace("data/index_pdbs/", "").replace(".pdb", "")
            if q_short == idx_short:
                p_self.append(p)
            else:
                p_nonself.append(p)

    # 5) 적합 품질 플롯 작성
    # --- 5) p-value 분포 시각화 ---
    if len(p_self) > 0 or len(p_nonself) > 0:
        plt.figure(figsize=(7,5))
        bins = np.logspace(-10, 0, 100)
        hist_self, edges = np.histogram(p_self, bins=bins, density=True)
        # non-self
        hist_nonself, _ = np.histogram(p_nonself, bins=bins, density=True)

        # bin 중심 계산
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

        # log-scale로 막대 직접 그리기
        plt.bar(bin_centers, hist_self, width=np.diff(edges),
            alpha=0.5, align='center', label='self', color='red', edgecolor='red')

        plt.bar(bin_centers, hist_nonself, width=np.diff(edges),
            alpha=0.5, align='center', label='non-self', color='blue', edgecolor='blue')
        plt.xscale('log')
        plt.xlim(1e-4, 1.0)
        plt.xlabel('p-value (log scale)')
        plt.ylabel('Density')
        plt.title('P-value Distribution: self vs non-self')
        plt.legend()
        plt.tight_layout()
        os.makedirs(PNG_DIR, exist_ok=True)
        out_png = os.path.join(PNG_DIR, "pvalue_distribution_self_vs_nonself.png")
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"[OK] Saved p-value distribution plot -> {out_png}")



def main():
    print("Starting Folddisco result to distribution analysis...")

    for file in os.listdir(RESULT_DIR):
        if FILE_INFO not in file:
            continue
        file_path   = os.path.join(RESULT_DIR, file)
        RESULT_FILE = os.path.join(RESULT_DIR, f"total_evalues_fitted2.txt")
        print(f"[INFO] Processing {file_path} ...")
        per_row_output(file_path, RESULT_FILE)
        print(f"[OK] Saved per-row e-values -> {RESULT_FILE}")
    

if __name__ == "__main__":
    with Pool(4) as p:
        main()

