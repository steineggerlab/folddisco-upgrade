# script to filter superfamilies that lack enough domains
INPUT=data/cath/cath-domain-list.txt
OUTPUT=data/cath/cath-domain-list-filtered.txt
CUTOFF=30 #Approxiamtely 25% of the superfamilies have 30 or more domains per superfamily (1646/6631)

awk -v cutoff="$CUTOFF" '
!/^#/ && NF >= 5 {
    key = $2 " " $3 " " $4 " " $5
    count[key]++
    lines[NR] = $0
    keys[NR] = key
}

END {
    for (i = 1; i <= NR; i++) {

        if (count[keys[i]] >= cutoff)
            print lines[i]
    }
}
' "$INPUT" > "$OUTPUT"
