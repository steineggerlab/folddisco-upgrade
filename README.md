# Folddisco-upgrade

This repository stores directories and files used to compute the e-values of Folddisco.
It consists relevant files from building a null set based on CATHdb to computing and fitting e-values of Folddisco matches.

## Directory: cath-db-v2
This directory contains the pipeline to collect pdb files from CATHdb, and filter them to consist a null set necessary for analysis.
The files are from CATH-Plus Version 4.3, and are clustered (nonredundant) data. Initial datasets are gathered in the data directory.
Superfamilies with too little domains were filtered (command/filtered.sh). Among the remaining superfamilies, one domain per superfamily were selected randomly (command/motif_select.sh).
Filtered pdbs, which would be the index of future Folddisco analysis, are gathered inside result/filtered_pdbs (command/pdb_organize.sh).

## Directory: foldseek-cluster-analysis
This directory contains the pipeline to impelment Foldseek-cluster to the pdbs, to check if the pdbs within the filtered group construct a cluster.
Since the pdbs should construct a null set, it is essential to verify that these pdbs have no relationship in terms of structure.
Foldseek-cluster was implemented by the script of command, which applies a grid search for coverage and e-value cutoffs.
Among the results shown, I have selected trial 15 (coverage = 0.5, e-value cutoff = 0.1) as the criteria. 63 pdbs were removed, with only 766 pdbs remaining.

## Directory: mmseqs-analysis
Apart from structure similarity, I also analyzed sequence similarity among the filtered pdbs, and ran mmseqs to verify any possible clusters among the pdbs.
MMseqs cluster was implemented by command/mmseqs_cluster.sh, which constructs fasta files from pdbs before clustering.
Only 4 pdbs were clustered and removed.

## Directory: folddisco-validation-result
The filtered and organized pdbs from the analysis beforehand were gathered to be data/index_pdbs. 
