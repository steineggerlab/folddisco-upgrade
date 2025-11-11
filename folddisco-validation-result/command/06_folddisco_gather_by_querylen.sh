#Script to gather folddisco outputs and classify them by query length
#!/bin/bash
#SBATCH --job-name=folddisco_gather_by_querylen # Job name
#SBATCH --ntasks=1 # Run on a single CPU
#SBATCH --time=01:00:00 # Time limit hrs:min:sec
#SBATCH --output=gather_by_querylen_%j.log # Standard output and error log

DATA_DIR='result/folddisco_results_raw'
TARGET_DIR='result/folddisco_results_analyses'
QUERY_INFO='result/folddisco_results_analyses/folddisco_result_sorted.txt'
FINAL_FILE="$TARGET_DIR"/"total_combined.txt"

first_file=$(find "$TARGET_DIR" -type f -name '*_combined.txt' | head -n 1)
{
    echo -e "query_num\t$(head -n 1 "$first_file")"

    # 모든 *_combined.txt 파일 순회
    find "$TARGET_DIR" -type f -name '*_combined.txt' | while read -r f; do
        file_name=$(basename "$f")  
        tail -n +2 "$f" | awk -v src="$file_name" '{print src "\t" $0}'
    done
} > "$FINAL_FILE"
