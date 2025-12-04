#!/bin/bash
#SBATCH --job-name=folddisco_gather_by_querylen
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=gather_by_querylen_%j.log

TARGET_DIR='result/folddisco_results_analyses'
FINAL_FILE="$TARGET_DIR/total_combined.txt"

# Pick first file to get header (length_* format)
first_file=$(find "$TARGET_DIR" -type f -name 'length_*_combined.txt' | sort | head -n 1)

if [[ -z "$first_file" ]]; then
    echo "No length_*_combined.txt found"
    exit 1
fi

# --- STEP 1: create unified header (matching total_combined.txt format) ---
# Remove col 8 (tm_score_strict), 11 (gdt_strict), 12 (rmsd_dup)

header=$(head -n 1 "$first_file" | awk '
BEGIN{OFS="\t"}
{
    # delete cols: 8, 11, 12  (awk is 1-based, so 9, 12, 13)
    $9=""; $12=""; $13="";
    gsub(/\t+/, "\t");
    sub(/^\t/, "");
    sub(/\t$/, "");
    print
}')

{
    echo -e "query_num\t${header}"

    # --- STEP 2: append rows from each length_X file after removing 8/11/12 ---
    find "$TARGET_DIR" -type f -name 'length_*_combined.txt' | sort | while read -r f; do
        qnum=$(basename "$f")
        tail -n +2 "$f" | awk -v Q="$qnum" '
        BEGIN{OFS="\t"}
        {
            # delete columns 8(tm_score_strict), 11(gdt_strict), 12(rmsd_dup)
            $9=""; $12=""; $13="";
            gsub(/\t+/, "\t");
            sub(/^\t/, "");
            sub(/\t$/, "");
            print Q, $0;
        }'
    done

} > "$FINAL_FILE"

echo "[OK] Created total_combined.txt with condensed 14-column schema: $FINAL_FILE"
