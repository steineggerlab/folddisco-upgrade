#Python script to analyze Folddisco results and make score distribution

import numpy as np
import matplotlib.pyplot as plt
import os
import math

RESULT_DIR = "result/folddisco_results_raw"
PNG_DIR = "result/analysis_plot"
NOKEY_FILE = "result/folddisco_results_stat/folddisco_nokey.txt"
RESULT_FILE = "result/folddisco_results_stat/folddisco_result_summary.txt"

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

def main():
    idf_score_dict = {}
    node_count_dict = {}
    print("Starting Folddisco result to distribution analysis...")
    for root, _, files in os.walk(RESULT_DIR):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r') as file:
                    if sum(1 for _ in file) < 2:
                        with open(NOKEY_FILE, 'a') as nokey_file:
                            nokey_file.write(f"{filename}\n")
                        continue
                    else:
                        idf_score_dict[filename] = []
                        node_count_dict[filename] = []
                    file.seek(0)  # Reset file pointer to the beginning
                    for line in file:
                        if line.startswith("id"):
                            continue
                        else:
                            cols = line.strip().split("\t")
                            idf_score_dict[filename].append(cols[2])
                            node_count_dict[filename].append(cols[1])
    for key, value in idf_score_dict.items():
        length = pdb_len(key.replace(".txt", ""))
        scores = [float(score) for score in value if score.replace('.', '', 1).isdigit()]
        if scores:
            mu, lam = evd_mle_full(scores)
            e = e_values(scores, mu, lam, M=len(scores))
            node_count = node_count_dict[key]
            with open(RESULT_FILE, 'a') as res_file:
                res_file.write(f"{key}\t{mu:.3f}\t{lam:.3f}\t{e[0]}\t{min(e)}\t{length}\t{max(node_count)}\n")
            
        else:
            with open(NOKEY_FILE, 'a') as nokey_file:
                nokey_file.write(f"{key}\n")

if __name__ == "__main__":
    main()
