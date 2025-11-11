#Python script to check e-value calibration of Folddisco results

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from math import exp
import os

FILE_INFO = "result/folddisco_results_analyses/evalues_stat2.txt"
PNG_DIR = "result/analysis_plot"

df = pd.read_csv(FILE_INFO, sep="\t")
x = df["e_value"].replace("NA", np.nan).dropna().astype(float)

# 1) Obs/Theory 곡선
thr = np.array([0.01,0.05,0.1,0.5,1,2,5,10])
obs = np.array([(x <= t).mean() for t in thr])
theo = 1 - np.exp(-thr)
ratio = obs/theo
for t,o,r in zip(thr, obs, ratio):
    print(f"t={t:>4}: obs={o:.4f} theo={1-exp(-t):.4f} ratio={r:.3f}")

# 2) PP-plot (uniform test)
p = 1 - np.exp(-x)       # p = 1 - e^{-E}
plt.figure(figsize=(4,4))
plt.plot(sorted(p), np.linspace(0,1,len(p)), '.')
plt.plot([0,1],[0,1],'r--')
plt.xlabel("Theoretical quantiles")
plt.ylabel("Empirical quantiles")
plt.title("PP plot of E-values")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PNG_DIR, f"PP_plot_10.png"), dpi=200)  
plt.close()
print("ppplot")               


# 3) 히스토그램
plt.figure(figsize=(4, 4))
plt.xlabel("E-value")
plt.ylabel("Density")
plt.title("Empirical E-value distribution")
plt.hist(x, bins=np.linspace(0, 1, 100), density=True, label="E-values")
plt.xlim(0, 1)               # x축 한계 설정
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PNG_DIR, "evalue_hist_10.png"), dpi=200)
plt.close()
print("histogram saved -> evalue_hist_10.png")
