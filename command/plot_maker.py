import matplotlib.pyplot as plt
import os

FILE = "result_whole/folddisco_results_stats/folddisco_result_summary_final.txt"

values = []

# --- 텍스트 파일 읽기 ---
with open(FILE) as f:
    for line in f:
        parts = line.strip().split("\t")   # 공백 기준 split, 탭이면 split("\t")
        if len(parts) < 4:
            continue
        
        try:
            v = float(parts[3])        # 4번째 열 (index=3)
        except ValueError:
            continue                   # 헤더거나 숫자가 아닐 경우 skip
        
        if v <= 32:
            values.append(v)

# --- bar plot ---
plt.figure(figsize=(12,4))
plt.bar(values, range(len(values)))
plt.xlabel("Length of query residues")
plt.ylabel("Count")
plt.title("Distribution for lengths of query residues")
plt.tight_layout()

plt.legend()
plt.tight_layout()

out_png = os.path.join(f"result_whole/folddisco_results_stats/length_distribution.png")
plt.savefig(out_png, dpi=300)
plt.close()

