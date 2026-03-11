# Folddisco-upgrade

This repository contains the end-to-end pipeline used to derive calibrated E-values for Folddisco matches. The core idea is to build a structure/sequence “null” background from CATH domains that are as unrelated as possible, then run Folddisco within a controlled query/index setup so that:
Unrelated pairs produce mostly low similarity scores (background / null distribution).
Within-superfamily pairs (true positives) produce higher similarity scores (signal).
This yields separable score distributions, enabling us to fit an E-value model for Folddisco match statistics.

(Data files including pdb files are to be released separately)

High-level pipeline (summary)
1. Construct a large CATH-derived index set
Gather and filter CATH-Plus v4.3 nonredundant domain PDBs to form an initial index set.
Result: N = 10547 domains.

2. Enforce “null-like” unrelatedness using structure clustering (Foldseek)
Run Foldseek clustering on the filtered set and remove any non-trivial structural clusters (i.e., domains that appear structurally related under chosen cutoffs).
Result: remove 63 clusters (trial 15 criteria).

3. Enforce unrelatedness using sequence clustering (MMseqs2)
Run MMseqs2 clustering on sequences extracted from the same set to catch residual sequence similarity clusters.
Result: remove 4 additional clustered domains.

4. Create representative “query anchors” and expand within-superfamily queries
From the remaining “non-cluster” candidates, build subgroups where each subgroup contains:
 - one non-cluster candidate domain (anchor), and
 - at least one additional domain from the same CATH superfamily (to provide positives).
Extract query sequences with Foldmason and apply quality filters (dominant-residue threshold, remove too-short queries, etc.).
Result: 372 valid subgroups with 1719 total query domains.

5. Run Folddisco and fit E-value parameters (length-aware → unified model)
Run Folddisco on query vs index matches, aggregate match statistics by query length (2–32), fit parameters, then validate.
Final workflow produces a single E-value function that can be applied across lengths (after calibration).


## Directory: 01-cath-data-collection
Pipeline to collect and filter CATH domains and produce the initial large index set.

Input: CATH-Plus v4.3 nonredundant domain data (in data/)

Steps:
Filter out superfamilies with too few domains (command/filtered.sh)
Randomly select one domain per remaining superfamily (command/motif_select.sh)
Organize PDB files (command/pdb_organize.sh)

Output: result/filtered_pdbs/ (N = 10547)

## Directory: 02-foldseek-cluster-analysis
Structural redundancy / similarity removal using Foldseek clustering.

Goal: ensure the “null” pool does not contain obvious structural relatives

Method: grid search over coverage + E-value cutoffs
Selected configuration in result/clusters
coverage = 0.5, e-value cutoff = 0.1

Outcome: remove domains participating in 63 non-self clusters

## Directory: 03-mmseqs-analysis
Sequence redundancy / similarity removal using MMseqs2 clustering.

Goal: remove remaining sequence-related domains that may survive structural filtering

Method: build FASTA from PDBs → run MMseqs clustering (command/mmseqs_cluster.sh)

Outcome: 4 domains clustered and removed
Resulting candidate pool: N = 657 “non-cluster” domains (used to form query representatives)

## Directory: 04-folddisco-analysis-preparation/
Data:
data/non_cluster_pdbs/ : filtered candidate anchors (N = 657)
data/index_pdbs/ : full index (N = 10547)

Steps:
1. Build “subgroups”: one candidate anchor + same-superfamily domains from the index
(command/00_gather_noncluster_pdbs_list.py)
Result: 494 subgroups with 1976 query domains

2. Extract query sequences with Foldmason and apply quality filters
 - dominant residue cutoff: 66%
 - remove queries of length 0–1
 - remove distant / unsuitable queries(command/01-02)
Result: 372 subgroups with 1719 query domains

3. Extract domain metadata and other annotations (command/03-04)


## Directory: folddisco-updated-validation
Main Folddisco run + score aggregation + E-value fitting + validation.

Steps:
1. Run Folddisco on constructed query/index sets (command/01, 01.5, 01.55)
2. Classify and aggregate results by query length; consolidate stats (command/02, 03)
3. Fit E-value parameters for query lengths 2–32 (command/04)
4. Derive a single unified E-value function (length-independent at inference time) and compute E-values for all matches (command/05)
5. Validate fit quality and calibration (command/06-08)

Fitting notes (important):
Multiple fitting schemes were explored (especially to reduce the fitted mu, which strongly affects calibration). These experiments live under command/00.
The finalized scheme is based on:
result_expanded/result_expanded4/
(This is the version you consider the “production” calibration.)
