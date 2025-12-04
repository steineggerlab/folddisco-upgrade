import csv

input_file = "data/chain_residue_list.txt"
output_file = "data/chain_residue_list_mismatch.txt"

with open(input_file) as fin, open(output_file, "w") as fout:
    for line in fin:
        parts = line.strip().split("\t")
        if len(parts) < 3:
            fout.write(line)
            continue
        
        col3 = parts[2]
        items = col3.split(",")           # ['A46','A50','A76', ...]
        new_items = [f"{x}:X" for x in items]
        parts[2] = ",".join(new_items)    # A46:X,A50:X,...

        fout.write("\t".join(parts) + "\n")
