#Script to gather folddisco outputs and classify them by query length
#!/bin/bash
#SBATCH --job-name=folddisco_gather_by_querylen # Job name
#SBATCH --ntasks=1 # Run on a single CPU
#SBATCH --time=01:00:00 # Time limit hrs:min:sec
#SBATCH --output=gather_by_querylen_%j.log # Standard output and error log

DATA_DIR='result/folddisco_results_raw'
TARGET_DIR='result/folddisco_results_analyses'
QUERY_INFO='result/folddisco_results_analyses/folddisco_result_sorted.txt'
TEMP_FILE="$TARGET_DIR"/"total_combined_temp.txt"
FINAL_FILE="$TARGET_DIR"/"total_combined.txt"

awk 'BEGIN{OFS="\t"} {
    $10=""; $13=""; $14="";
    gsub(/\t+/, "\t");
    sub(/^\t/, "");
    sub(/\t$/, "");
    print
}' "$TEMP_FILE" > "$FINAL_FILE"
