tar -xvzf cath-dataset-nonredundant-S20.pdb.tgz
cut -f1 null_pairs.tsv > domain1.txt
cut -f2 null_pairs.tsv > domain2.txt
cat domain1.txt domain2.txt | sort | uniq > null_ids.txt
sed -E 's/^>cath\|[^|]*\|([a-zA-Z0-9]+)\/.*/>\1/' cath-domain-seqs-S60.fa > cath-domain-seqs-S60.clean.fa
seqtk subseq cath-domain-seqs-S60.clean.fa null_ids.txt > null_pairs.fasta


