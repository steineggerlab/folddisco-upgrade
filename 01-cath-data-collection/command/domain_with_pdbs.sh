#script to select domains that have a coresponding PDB file

DIR=data/dompdb
INPUT=data/cath/cath-domain-list-filtered.txt
PDBLIST=result/pdb_ids.txt
OUTPUT=result/domain-list-pdbexists.txt

find "$DIR" -type f -exec basename {} \; \
| sed 's/\.[^.]*$//' \
| sort -u > "$PDBLIST"

awk 'NR==FNR {ids[$1]; next} $1 in ids' "$PDBLIST" "$INPUT" > "$OUTPUT" 

