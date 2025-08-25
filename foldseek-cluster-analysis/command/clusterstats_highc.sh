#Script to generate cluster statistics for Foldseek clustering trials

OUTPUT="result/foldseek_cluster_stats.tsv"

\echo -e "Coverage\te=0.001\te=0.05\te=0.1\te=0.5" > "$OUTPUT"

coverages=(0.9 0.8 0.7 0.6 0.5)

e0001_trials=(1 2 3 4 5)
e005_trials=(6 7 8 9 10)
e01_trials=(11 12 13 14 15)
e05_trials=(16 17 18 19 20)

for i in {0..4}; do
    cov=${coverages[$i]}
    file1="result/foldseek-trial${e0001_trials[$i]}/trial${e0001_trials[$i]}_cluster.tsv"
    file2="result/foldseek-trial${e005_trials[$i]}/trial${e005_trials[$i]}_cluster.tsv"
    file3="result/foldseek-trial${e01_trials[$i]}/trial${e01_trials[$i]}_cluster.tsv"
    file4="result/foldseek-trial${e05_trials[$i]}/trial${e05_trials[$i]}_cluster.tsv"

    if [ -f "$file1" ]; then
        total1=$(wc -l < "$file1")
        neq1=$(awk '$1 != $2' "$file1" | wc -l)
        percent1=$(awk -v a="$neq1" -v b="$total1" 'BEGIN { if(b>0) printf "%.2f%%", a/b*100; else print "NA" }')
        res1="$neq1 ($percent1)"
    else
        res1="NA"
    fi

    if [ -f "$file2" ]; then
        total2=$(wc -l < "$file2")
        neq2=$(awk '$1 != $2' "$file2" | wc -l)
        percent2=$(awk -v a="$neq2" -v b="$total2" 'BEGIN { if(b>0) printf "%.2f%%", a/b*100; else print "NA" }')
        res2="$neq2 ($percent2)"
    else
        res2="NA"
    fi

    if [ -f "$file3" ]; then
        total3=$(wc -l < "$file3")
        neq3=$(awk '$1 != $2' "$file3" | wc -l)
        percent3=$(awk -v a="$neq3" -v b="$total3" 'BEGIN { if(b>0) printf "%.2f%%", a/b*100; else print "NA" }')
        res3="$neq3 ($percent3)"
    else
        res3="NA"
    fi

    if [ -f "$file4" ]; then
        total4=$(wc -l < "$file4")
        neq4=$(awk '$1 != $2' "$file4" | wc -l)
        percent4=$(awk -v a="$neq4" -v b="$total4" 'BEGIN { if(b>0) printf "%.2f%%", a/b*100; else print "NA" }')
        res4="$neq4 ($percent4)"
    else
        res4="NA"
    fi

    echo -e "${cov}\t${res1}\t${res2}\t${res3}\t${res4}" >> "$OUTPUT"
done