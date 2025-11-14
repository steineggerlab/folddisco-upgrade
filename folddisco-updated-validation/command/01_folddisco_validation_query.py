#Python script to run FoldDisco queries on a set of PDB files and save the results.

import os, subprocess
from collections import defaultdict

DATA_DIR='data/classified_pdbs'
DOMAIN_INFO='domain_list.txt'
QUERY_INFO='data/chain_residue_list.txt'
INDEX_DIR='index/index_pdbs_folddisco' 
RESULT_DIR='result/folddisco_results_raw'
INFO_LIST_FILE='data/folddisco_info_list.txt'
FOLDDISCO_COMMAND_FILE = "command/folddisco_commands.txt"
FOLDDISCO = "folddisco"  # Path to foldmason executable
BIGNUM = 999999999999
HEADER = 'id\tnode_count\tidf_score_per_match\trmsd\tmatching_residues\tkey\ttm_score\ttm_score_strict\tgdt_ts\tgdt_ha\tgdt_strict\trmsd\tchamfer_distance\thausdorff_distance\tquery_residues'

os.makedirs(RESULT_DIR, exist_ok=True)

for file in os.listdir(DATA_DIR):
    if DOMAIN_INFO in os.listdir(os.path.join(DATA_DIR, file)):
        dir_path = os.path.join(DATA_DIR, file)
        relative_path = os.path.relpath(dir_path, DATA_DIR)

        for pdb in os.listdir(dir_path):
            if not pdb.endswith('.pdb'):
                continue
            pdb_id = pdb.replace('.pdb', '')
            query_file = os.path.join(dir_path, pdb_id)
            OUTPUT_DIR = os.path.join(RESULT_DIR, relative_path)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            output_file = os.path.join(OUTPUT_DIR, f"output_{pdb_id}.txt")
            residues = []
            for line in open(QUERY_INFO, 'r'):
                parts = line.strip().split("\t")
                if parts[0] == relative_path and parts[1] == pdb_id:
                    if len(parts) > 2:
                        residues = parts[2].split(",")
                        break
                    else:
                        dummy = 'A'+ str(BIGNUM)
                        residues = [dummy]
                        break
            
            with open(INFO_LIST_FILE, 'a') as f:  
                f.write(f"{os.path.join(dir_path, pdb)}\t{','.join(residues)}\t{output_file}\n")

command = (
    f"{FOLDDISCO} query -i {INDEX_DIR} "
    f"-q {INFO_LIST_FILE} -t 128 -v --sort-by idf --header"
    )
with open(FOLDDISCO_COMMAND_FILE, 'w') as f:  
    f.write(f"{command}\n")

print("Executing folddisco commands.")
os.system(f"bash {FOLDDISCO_COMMAND_FILE}")
print("Folddisco finished.")