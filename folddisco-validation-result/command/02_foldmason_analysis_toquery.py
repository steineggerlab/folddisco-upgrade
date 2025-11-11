#Python script to analyze FoldMason validation results

from Bio import AlignIO, SeqIO
from collections import Counter
import numpy as np
import os

DATA_DIR = "data/foldmason_1"
RESULT_DIR = "data/query_sequences_1"
FASTA_FILE = "result_aa.fa"
CUTOFF_COLUMN = 0.66
CUTOFF_COLUMN_MIN = 0
CUTOFF_QUERY_MIN = 1

# Read MSA from a given file path
def read_alignment(file_path):
    try:
        alignment = AlignIO.read(f"{file_path}/{FASTA_FILE}", "fasta")
        return alignment
    except Exception as e:
        return 0
    
def get_header(file_path):
    try:
        ids = [record.id for record in SeqIO.parse(f"{file_path}/{FASTA_FILE}", "fasta")]
        return ids
    except Exception as e:
        return 0

# Function to convert alignment to a matrix of characters (insertions ignored)
def align_to_matrix(msa):
    alignment_matrix = []
    num_seqs = len(msa)
    for record in msa:
        seq_row = []
        for res in str(record.seq):
            if res == '-':
                seq_row.append('-')
            elif res.islower():
                seq_row.append('-')
            else:
                seq_row.append(res.upper())
        alignment_matrix.append(seq_row)
    alignment_matrix = np.array(alignment_matrix)

    return alignment_matrix, num_seqs

# Identify column indices of conserved columns
def column_maker(alignment_matrix, num_seqs):
    conserved_columns = []
    freq_columns = []
    for col_idx in range(alignment_matrix.shape[1]):
        column = alignment_matrix[:, col_idx]
        unique, counts = np.unique(column, return_counts=True)
        max_freq = max(counts) / num_seqs
    # If the most frequent residue meets the threshold, it's a conserved column
        if max_freq >= CUTOFF_COLUMN and '-' not in unique:
            conserved_columns.append(col_idx)
            freq_columns.append(max_freq)
    return conserved_columns, freq_columns

def extract_query(msa, conserved_columns):
    query_sequence = []
    for col_idx in conserved_columns:
        column = msa[:, col_idx]
        most_common_residue = Counter(column).most_common(1)[0][0]
        query_sequence.append(most_common_residue)
    return query_sequence

def main():
    for filename in os.listdir(DATA_DIR):
        msa = read_alignment(os.path.join(DATA_DIR, filename))
        pdb_list = get_header(os.path.join(DATA_DIR, filename))
        alignment_matrix, num_seqs = align_to_matrix(msa)
        conserved_columns, frequency_columns = column_maker(alignment_matrix, num_seqs)

        if len(conserved_columns) <= CUTOFF_COLUMN_MIN:
            continue
        else:
            query_sequence = extract_query(msa, conserved_columns)
            if len(query_sequence) <= CUTOFF_QUERY_MIN:
                continue
            if not os.path.exists(RESULT_DIR):
                os.makedirs(RESULT_DIR)
            with open(os.path.join(RESULT_DIR, f"{filename}_query.txt"), "w") as f:
                f.write(f">{filename}_query\n")
                f.write("PDBs: " + ", ".join(pdb_list) + "\n")
                f.write("".join(query_sequence) + "\n")
                f.write("".join(str(conserved_columns)) + "\n")
                f.write("".join(str(frequency_columns)) + "\n")
        

if __name__ == "__main__":
    main()