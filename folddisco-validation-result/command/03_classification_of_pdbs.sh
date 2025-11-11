#Script to cluster pdbs by CAT and prepare running Folddisco queries on them

foldmason_result_dir='data/query_sequences_1'
pdbs_dir='data/index_pdbs'
output_dir='data/classified_pdbs/pdbs_'

for file in $foldmason_result_dir/*_query.txt; do
    cat_id=$(basename "$file" _query.txt)
    outdir="${output_dir}${cat_id}"
    mkdir -p "$outdir"
    line=$(awk 'NR==2{gsub(/^PDBs:[ \t]*/,""); print}' $file)
    pdb_list=$(echo $line | tr ',' '\n')
    for pdb in $pdb_list; do
        cp ${pdbs_dir}/${pdb}.pdb ${output_dir}${cat_id}/
    done
    awk 'NR>=3 && NR<=5' $file > ${output_dir}${cat_id}/domain_list.txt
done
