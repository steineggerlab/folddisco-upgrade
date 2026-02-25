# Folddisco-upgrade

This repository stores directories and files used to compute the e-values of Folddisco.
It consists relevant files from building a null set based on CATHdb to computing and fitting e-values of Folddisco matches.
The main idea is that, running Folddisco with psbs inside a null set consisting of pdbs with low similarity would result in low similarities for non related pdbs, while high similarities for self pdbs or pdbs within the same homology group.
This would lead to two distinct distributions of score, which could lead to computation of e-values for Folddisco matches.
The summarized pipeline is as follows:
1. Gather and filter pdbs from CATHdb to construct a query pdb set with pdbs that have low similarity.
2. Construct an index that contains pdbs within the query, and pdbs within the same superfamily with the query.
3. Extract query sequences using MSA results from Foldmason.
4. Run Folddisco and compute appropritate fitting scheme for e-values


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
After that, I implemented Foldmason to extract query sequences from each subgroup. Using a cutoff of 66% as a dominant residue for the query sequence, and excluding 0, 1 residue queries, distant queries, a total of 372 subgroups with 1716 pdbs were left for Folddisco analysis (command/01-02). Necessary information such as domain info were extracted from the query (command/03-04). The remaining commands are duplicated in the next directory.

## Directory: folddisco-updated-validation
I ran Folddisco based on the gathered query and index(command/01, 01.5, 01.55). After that, the results were classified by query length, and were gathered(command/02, 03). Parameters for e-value fitting were computed for queryies ranging in length 2-32(command/04). Based on these results, a single function computing e-values (regardless of query length) computes all e-values for all matches (command/05), followed by analysis and fitting validation (command/06-08).
To fit e-values properly (especially, to lower the mu necessary for e-value fitting), various schemes were attempted(command/00). The finalized version is based on result_expanded/result_expanded4.
