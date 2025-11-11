#Python script to implement foldmason to make alignments from  null set(non-cluster pdbs)

from pathlib import Path
import os, subprocess
from collections import defaultdict

INDEX_LIST = "data/domain-list-index.txt"
NULL_LIST = "data/domain-list-noncluster.txt"
INDEX_MOD = "data/index_list.txt"
NULL_MOD = "data/null_list.txt"
INDEX_DIR = "data/index_pdbs"
NULL_DIR = "data/non_cluster_pdbs"
OUTPUT_DIR = "data/foldmason_1"
STAT_FILE = "data/subgroups.tsv"
TMP_DIR = "tmp"
FOLDMASON = "foldmason"  # Path to foldmason executable
FOLDMASON_COMMAND_FILE = "command/foldmason_commands.txt"
CUTOFF = 2  # Minimum number of PDBs to run foldmason

def list_to_row(input_list):
    rows = []
    with open(input_list, 'r') as infile:
        for line in infile:
            if line.startswith('ID') or not line.strip():
                continue
            parts = line.strip().split()
            rows.append({"ID":parts[0], "C":parts[1], "A":parts[2], "T":parts[3]})
    return rows

def cath_to_key(row):
    return (row ["C"], row["A"], row["T"])

def id_to_path(pid):
    return os.path.join(INDEX_DIR, pid + ".pdb")

def run(cmd, cwd=None):
    print("$", " ".join(cmd) + (f"   [cwd={cwd}]" if cwd else ""))
    subprocess.run(cmd, check=True, cwd=cwd)

def main():
    print("Preparing foldmason input")
    rows_index = list_to_row(INDEX_LIST)
    rows_null = list_to_row(NULL_LIST)   

    index_id_dict = defaultdict(list)
    null_id_dict = defaultdict(list)
    combined_dict = defaultdict(list)
    for r in rows_index:
        index_id_dict[cath_to_key(r)].append(r["ID"])
        
    for r in rows_null:
        null_id_dict[cath_to_key(r)].append(r["ID"])

    for k, v in index_id_dict.items():
        if k in null_id_dict.keys():
            combined_dict[k] = v
    with open(STAT_FILE, 'w') as f:
        f.write("C\tA\tT\tNum_PDBs\n")
    for k, v in combined_dict.items():
        with open(STAT_FILE, 'a') as f:
            f.write(f"{k[0]}\t{k[1]}\t{k[2]}\t{int(len(v))}\n")

    print(len(combined_dict.keys()), "CATs in null set")

    for key, value in combined_dict.items():

        cat_str = "_".join(key)                            
        cat_out = Path(OUTPUT_DIR) / cat_str                
        cat_tmp = Path(TMP_DIR) / cat_str                   
    
        pdb_files = [id_to_path(vv) for vv in value if os.path.isfile(id_to_path(vv))]
        if len(pdb_files) < CUTOFF:
            print("Skipping key with less than", CUTOFF, "PDBs:", key)
            continue
        
        os.makedirs(cat_out, exist_ok=True)
        os.makedirs(cat_tmp, exist_ok=True)
        
        files = " ".join(pdb_files)
        command = (
        f"{FOLDMASON} easy-msa {files} "
        f"{cat_out}/result {cat_tmp} --report-mode 2"
        )
        with open(FOLDMASON_COMMAND_FILE, 'a') as f:  
            f.write(f"{command}\n")

    print("Executing foldmason commands.")
    os.system(f"bash {FOLDMASON_COMMAND_FILE}")
    print("Foldmason finished.")


if __name__ == "__main__":
    main()