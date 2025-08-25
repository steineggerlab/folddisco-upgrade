#script to select domains randomly from each distinct superfamily
[ -d "result/filtered_pdbs" ] && rm -rf result/filtered_pdbs
mkdir -p result/filtered_pdbs

INPUT=result/domain-list-pdbexists.txt
DATA=data/dompdb
DEST=result/filtered_pdbs
INFO=result/domain_info.tsv
OUTPUT=result/domain-list-selected.txt

awk '{print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$11}' "$INPUT" > "$INFO"

#domains are grouped as superfamily (distinguishted by CAT)
awk '{key = $2 "_" $3 "_" $4
      print > ("tmp_" key ".txt")
}' "$INFO"

# one domain per superfamily is randomly selected
for f in tmp_*.txt; do
    shuf -n 1 "$f"
done | cat > "$OUTPUT"

rm tmp_*.txt

#copying pdb files correspondent to the selected domainst
cut -f1 $OUTPUT | sort -u | \
while read id; do
    src="$DATA/${id}.pdb"
    dst="$DEST/${id}.pdb"
    [ -f "$src" ] && cp "$src" "$dst"
done
