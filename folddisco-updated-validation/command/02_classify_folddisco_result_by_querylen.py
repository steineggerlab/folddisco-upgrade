#Python script to classify outputs from Folddisco validation by query length

import glob
import os
from collections import defaultdict
DATA_DIR = "result/folddisco_results_raw"
RESULT_DIR = "result/folddisco_results_analyses"
DATA_FILE = "result/folddisco_results_stats/folddisco_result_summary.txt"

header = "source\tid\tnode_count\tidf_score_per_match\trmsd\tmatching_residues\tkey\ttm_score\ttm_score_strict\tgdt_ts\tgdt_ha\tgdt_strict\trmsd\tchamfer_distance\thausdorff_distance\tquery_residues"

os.makedirs(RESULT_DIR, exist_ok=True)

length_categories = defaultdict(list)

with open(DATA_FILE) as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.strip().split("\t")
        query_id = parts[0]
        query_length = int(parts[2])

        length_categories[query_length].append(query_id)

for length, query_ids in length_categories.items():
    if length > 9 :
        continue
    length_dir = os.path.join(RESULT_DIR, f"length_{length}")
    os.makedirs(length_dir, exist_ok=True)

    combined_path = os.path.join(RESULT_DIR, f"length_{length}_combined.txt")

    with open(combined_path, "w") as combined_out:
        combined_out.write(f"{header}\n")
        for query_id in query_ids:
            pattern = os.path.join(DATA_DIR, "**", f"{query_id}")
            matches = glob.glob(pattern, recursive=True)

            if not matches:
                continue

            base_name = os.path.basename(matches[0])
            dest_path = os.path.join(length_dir, base_name)

            with open(matches[0]) as src_file:
                content = src_file.read()
            with open(dest_path, "w") as dest_file:
                dest_file.write(content)

            combined_out.write(f"{base_name}\t{content}\n")

    

    for file in os.listdir(length_dir):
        full_path = os.path.join(length_dir, file)
        if full_path.endswith(".txt"):
            os.remove(full_path)


