# Script to perform sliding window analysis for cathdb dataset

# pass parameters from command line
# window size: default 10
# step size: default 5
# Usage: bash sliding_window.sh [window_size] [step_size] [coverage] [evalue]
WINDOW_SIZE=${1:-10}
STEP_SIZE=${2:-5}
COVERAGE=${3:-0.9}
EVALUE=${4:-0.001}

# Making fasta files from pdb files
for pdb in data/pdbs/*.pdb; do
    pdb_tofasta "$pdb" > "data/fastas/$(basename "$pdb" .pdb).fasta"
done
cat data/fastas/*.fasta > result/all_seqs.fasta

# Making Sliding window fragments (defaults are as follows: window: 10, step: 5)
rm -f result/windows_*.fasta
rm -f windows_db*
rm -f windows_cluster*
for win in $WINDOW_SIZE; do
  awk -v w=$win -v s=$STEP_SIZE '
  /^>/ {header=$0; next}
  {
      seq=$0
      for(i=1; i<=length(seq)-w+1; i+=s) {
          print header "_win" w "_" i
          print substr(seq, i, w)
      }
  }' result/all_seqs.fasta > result/windows_all.fasta
done

# Making MMSeqs2 DB + clustering : TBUD
mmseqs createdb result/windows_all.fasta windows_db
mmseqs cluster windows_db windows_cluster tmp_dir -c $COVERAGE --min-seq-id 0.2 -e $EVALUE
mmseqs createtsv windows_db windows_db windows_cluster result/windows_clusters_"$EVALUE"_"$COVERAGE"_"$WINDOW_SIZE"_"$STEP_SIZE".tsv
