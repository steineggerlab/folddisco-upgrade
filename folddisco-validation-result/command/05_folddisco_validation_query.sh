#Script to run FoldDisco queries on a set of PDB files and save the results with headers. 

#!/bin/bash 
#SBATCH --job-name=folddisco_job # Job name 
#SBATCH --ntasks=64 # Run on a single CPU 
#SBATCH --time=06:00:00 # Time limit hrs:min:sec 
#SBATCH --output=serial_test_%j.log # Standard output and error log 

DATA_DIR='data/classified_pdbs'
DOMAIN_INFO='domain_list.txt'
QUERY_INFO='data/chain_residue_list.txt'
INDEX_DIR='index/index_pdbs_folddisco' 
RESULT_DIR='result/folddisco_results_raw' 
HEADER=$'id\tnode_count\tidf_score\trmsd\tmatching_residues\tkey\tquery_residues'

for file in "$DATA_DIR"/*/"$DOMAIN_INFO"; do
    dir="$(dirname "$file")"                     
    cat_id="$(basename "$dir")"
    outdir="$RESULT_DIR/$cat_id"
    mkdir -p "$outdir"

    for pdb in $dir/*.pdb; do 
        pdb_id=$(basename "$pdb" .pdb)
        Q=$(awk -v id="$pdb_id" '
            $2 == id {
            for (i=3; i<=NF; i++) {
              if ($i != "NA") {
                printf "%s ", $i
              }
            }
        }' "$QUERY_INFO")
        output_file="$outdir/output_$pdb_id.txt" 
        folddisco query -p $dir/$pdb_id.pdb -i $INDEX_DIR -q $Q -t 8 -v --sort-by-score > "$RESULT_DIR/tmp.txt" 
        { 
          echo -e "$HEADER"; 
          cat "$RESULT_DIR/tmp.txt"; 
        } > "$output_file"
      done 
      rm -f "$RESULT_DIR/tmp.txt" 
done
