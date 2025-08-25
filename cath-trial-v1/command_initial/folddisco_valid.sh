INPUT_TSV="null_pairs.tsv"
STRUCT_DIR="filtered_structures"
OUTPUT_TSV="folddisco_results.tsv"

echo -e "Query\tTarget\tNodeCount\tAvgIDF\tRMSD" > "$OUTPUT_TSV"

FOLDDISCO_BIN="folddisco"

while IFS=$'\t' read -r query target; do
    query_pdb="$STRUCT_DIR/${query}"
    target_pdb="$STRUCT_DIR/${target}"

    if [[ ! -f "$query_pdb" || ! -f "$target_pdb" ]]; then
        echo "Skipping missing file: $query or $target"
        continue
    fi

    result=$($FOLDDISCO_BIN "$query_pdb" "$target_pdb" 2>/dev/null)

    if [[ -z "$result" ]]; then
        echo "No result for: $query vs $target"
        continue
    fi

    metrics=$(echo "$result" | awk 'NR==2 {print $2"\t"$3"\t"$4}')
    echo -e "$query\t$target\t$metrics" >> "$OUTPUT_TSV"
done < "$INPUT_TSV"
