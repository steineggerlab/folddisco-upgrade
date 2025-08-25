#script to select domains randomly from each distinct superfamily

INPUT=data/cath/cath-domain-list-filtered.txt
INFO=result/domain_info.tsv
OUTPUT=result/domain-list-selected.txt

awk '{print $1"\t"$2"\t"$3"\t"$4"\t"$5}' "$INPUT" > "$INFO"

#domains are grouped as superfamily (distinguishted by CAT)
awk '{key = $2 "_" $3 "_" $4
      print > ("tmp_" key ".txt")
}' "$INFO"

# one domain per superfamily is randomly selected
for f in tmp_*.txt; do
    shuf -n 1 "$f"
done | cat > "$OUTPUT"

rm tmp_*.txt
