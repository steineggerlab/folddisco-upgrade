#Script to leave only domains that are included in non-cluster pdbs in the list of domains
import os
import pandas as pd

DATA_DIR = "data/non_cluster_pdbs"
LISTFILE = "data/domain-list-index.txt"
RESULTFILE = "data/domain-list-noncluster.txt"

header = ["ID", "Class", "Architecture", "Topology(Folds)", "Hierarchy(Superfamily)"]

pdb_ids = {os.path.splitext(f)[0] for f in os.listdir(DATA_DIR) if f.endswith(".pdb")}

rows = []
with open(LISTFILE, "r") as f:
    for line in f:
        if line.startswith("ID") or not line.strip():
            continue
        parts = line.strip().split('\t')
        pdb_id = parts[0]
        if pdb_id in pdb_ids:
            rows.append(parts)

df = pd.DataFrame(rows, columns=header)
df = df.sort_values(by=["Class", "Architecture", "Topology(Folds)"], 
                    key=lambda col: pd.to_numeric(col, errors="coerce"))

df.to_csv(RESULTFILE, sep="\t", index=False)
print(f"Saved filtered list to {RESULTFILE}")