N_PAIRS=100000
OUTPUT=null_pairs.tsv

grep -v '^#' cath-domain-list.txt | awk '{print $1"\t"$4}' > domain_topology.tsv
cut -f1,2 domain_topology.tsv > domain_topo.txt

cut -f1 domain_topo.txt > domain_ids.txt

# null_pairs 생성
awk -v n="$N_PAIRS" '
BEGIN {
    # Read domain → topology mapping
    while ((getline < "domain_topo.txt") > 0) {
        topo[$1] = $2
        ids[++N] = $1
    }
    srand()

    count = 0
    while (count < n) {
        i = int(rand() * N) + 1
        j = int(rand() * N) + 1

        id1 = ids[i]
        id2 = ids[j]

        if (id1 == id2) continue
        if (topo[id1] == topo[id2]) continue

        print id1 "\t" id2
        count++
    }
}' > "$OUTPUT"


