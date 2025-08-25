#script to copy pdb files correspondent to the selected domains

mkdir -p result/filtered_pdbs

INPUT=result/domain-list-selected.txt
DATA=data/dompdb
DEST=result/filtered_pdbs

cut -f1 $INPUT | sort -u | \
while read id; do
    src="$DATA/${id}.pdb"
    dst="$DEST/${id}.pdb"
    [ -f "$src" ] && cp "$src" "$dst"
done
