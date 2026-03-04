import os
import glob

TARGET_DIR = "result/folddisco_results_analyses"
FINAL_FILE = os.path.join(TARGET_DIR, "total_combined.txt")

pattern = os.path.join(TARGET_DIR, "length_*_combined.txt")
files = sorted(glob.glob(pattern))

if not files:
    raise SystemExit(f"No length_*_combined.txt found in: {TARGET_DIR}")

# Read header from first file
with open(files[0], "r") as f:
    header_line = f.readline().rstrip("\n")

if not header_line:
    raise SystemExit(f"First file has empty header: {files[0]}")

os.makedirs(TARGET_DIR, exist_ok=True)

with open(FINAL_FILE, "w") as out:
    # Unified header
    out.write("query_num\t" + header_line + "\n")

    # Append rows from each file
    for fp in files:
        qnum = os.path.basename(fp)
        with open(fp, "r") as f:
            # skip per-file header
            _ = f.readline()
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                out.write(f"{qnum}\t{line}\n")

print(f"[OK] Created: {FINAL_FILE}")