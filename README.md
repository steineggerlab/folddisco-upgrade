# Folddisco-upgrade

This repository stores directories and files used to compute the e-values of Folddisco.
It consists relevant files from building a null set based on CATHdb to computing and fitting e-values of Folddisco matches.
The summarized pipeline is as follows:


## Directory: cath-db-v2
This directory contains the pipeline to collect pdb files from CATHdb, and filter them to consist a null set necessary for analysis.
The files are from CATH-Plus Version 4.3, and are clustered (nonredundant) data. Initial datasets are gathered in the data directory.
Superfamilies with too little domains were filtered (command/filtered.sh). Among the remaining superfamilies, one domain per superfamily were selected randomly (command/motif_select.sh).
The filtered pdbs are gathered inside result/filtered_pdbs (command/pdb_organize.sh).

## Directory: foldseek-cluster-analysis
This directory contains the pipeline to impelment Foldseek-cluster to the pdbs, to check if some pdbs within the filtered pdbs construct a non-self cluster.
Since the pdbs should construct a null set, it is essential to verify that these pdbs have no relationship in terms of structure.
Foldseek-cluster was implemented by the script of command, which applies a grid search for coverage and e-value cutoffs.
Among the results shown, I have selected trial 15 (coverage = 0.5, e-value cutoff = 0.1) as the criteria. Pdbs that consisted a total of 63 clusters were removed.

## Directory: mmseqs-analysis
Apart from structure similarity, I also analyzed sequence similarity among the filtered pdbs, and ran mmseqs to verify any possible clusters among the pdbs.
MMseqs cluster was implemented by command/mmseqs_cluster.sh, which constructs fasta files from pdbs before clustering.
Only 4 pdbs were clustered and removed. This left a total of 657 pdbs inside the filtered pdbs.

## Directory: folddisco-validation-result
The filtered and organized pdbs from the analysis beforehand were gathered to be data/non_cluster_pdbs (N=657). Pdbs that belong in the same superfamily with each filtered pdb are gathered to be data/index_pdbs (N=10547).
I first constructed a subgroup, which defines a group of pdbs that includes one filtered pdb, and at least one pdb from the index that belong in the same superfamily (command/00_gather_noncluster_pdbs_list.py). This left 494 subgroups with 1976 query pdbs.
After that, I implemented Foldmason to extract query sequences from each subgroup.
