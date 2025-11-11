#Python script to analyze Folddisco results and make score distribution

import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.stats import gumbel_r

RESULT_DIR = "result/folddisco_result"
PNG_DIR = "result/folddisco_plot"

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

def main():
    idf_score_dict = {}
    print("Starting Folddisco result to distribution analysis...")
    for filename in os.listdir(RESULT_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(RESULT_DIR, filename)
            with open(file_path, 'r') as file:
                if sum(1 for _ in file) < 2:
                    print(f"{filename} is empty or has insufficient data.")
                    continue
                else:
                    idf_score_dict[filename] = []
                file.seek(0)  # Reset file pointer to the beginning
                for line in file:
                    if line.startswith("id"):
                        continue
                    else:
                        cols = line.strip().split("\t")
                        idf_score_dict[filename].append(cols[2])
    for key, value in idf_score_dict.items():
        scores = [float(score) for score in value if score.replace('.', '', 1).isdigit()]
        if scores:
            mu, lam = evd_mle_full(scores)
            e = e_values(scores, mu, lam, M=len(scores))
            plot_distribution_score(scores, mu, lam,
                  title="IDF score fit", outfile = f"{PNG_DIR}/{key}_distribution.png", dpi=300)
            plot_distribution_evalue(e,
                  title="IDF evalue fit", outfile = f"{PNG_DIR}/{key}_evalue.png", dpi=300)
            print(f"Processed {key}: mu={mu:.3f}, lambda={lam:.3f}, e-value_self={e[0]}, e-value_min={min(e)}")
            
        else:
            print(f"No valid scores found in {key}")

if __name__ == "__main__":
    main()
