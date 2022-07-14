
import pandas as pd
import numpy as np
from pathlib import Path

from py import process
from utils import *
import sys

_parent_path = Path(__file__).parent

SVs = {
    0: NO_SV,
    1: DEL,
    2: DUP,
    3: INS,
    4: INV
}

def create_tsv_dataset(dataset_path):
    df = pd.read_csv(dataset_path, sep=";")
    sequences = df.NEW_SEQ.values
    labels = df.SV_TYPE.values
    processed_sequences = [preprocess_sequence(s) for s in sequences]

    new_df = pd.DataFrame({
            "sequence": processed_sequences,
            "label": labels
        })

    n = len(new_df)
    n_train = int(0.8 * n)

    df_train = new_df.iloc[:n_train]
    df_eval = new_df[n_train:]

    df_train.to_csv("fine_tune_data/train.tsv", sep="\t", index=False)
    df_eval.to_csv("fine_tune_data/eval.tsv", sep="\t", index=False)


def create_homogenous_dataset(N, ratios, chromosome_name, same_sequence):
    """
    Create a dataset from the same chromosome.
    N : length dataset
    ratios : ratios of SVs to apply, in alphabetical order, e.g NO_SV, DEL, DUP, INS, INV
    chromosome_name : name of the chromosome
    same_sequence : will perform SVs on the same sequence
    """

    # Normalize ratios
    ratios = ratios / np.sum(ratios)

    # Create name of file 
    file_name = f'data/chromosomes/{chromosome_name}.fa'

    # Initialize columns
    if same_sequence:
        seq = extract_sequence(file_name=file_name, N=1)
        original_sequences = np.full(N, seq[0])
    else:
        original_sequences = extract_sequence(file_name=file_name, N=N)
    
    # How many auxiliary sequence do we need?
    N_aux = int(ratios[3] * N)
    auxiliary_sequences = extract_sequence(file_name=file_name, N=N_aux)

    breakpoint_1 = np.full(N, None)
    breakpoint_2 = np.full(N, None)
    new_sequences = np.full(N, None)
    chromosome_used = np.full(N, chromosome_name)

    # Prepare SVs to perform
    sv_type = np.random.choice([0,1,2,3,4], size=N, p=ratios)

    # Produce rows of dataframe
    for i in range(N):
        seq = str(original_sequences[i])
        if sv_type[i] == 3: # if it's an insertion
            aux_seq = str(np.random.choice(auxiliary_sequences, size=1))
            new_seq, b1, b2, _ = SVs[sv_type[i]](seq, aux_seq)
        else:
            new_seq, b1, b2, _ = SVs[sv_type[i]](seq)

        breakpoint_1[i] = b1
        breakpoint_2[i] = b2
        new_sequences[i] = new_seq

    df = pd.DataFrame({
        "ORIGINAL_SEQ": original_sequences,
        "SV_TYPE": sv_type,
        "BP_1": breakpoint_1,
        "BP_2": breakpoint_2,
        "NEW_SEQ": new_sequences,
        "CHR_NAME": chromosome_used
    })

    return df

if __name__ == "__main__":
    
    N = int(sys.argv[1])
    ratios = np.array(sys.argv[2:7], dtype="float64")
    ratios = ratios / np.sum(ratios)
    chromosome_name = sys.argv[7]
    same_sequence = eval(sys.argv[8])

    df = create_homogenous_dataset(N=N, ratios=ratios, chromosome_name=chromosome_name, same_sequence=same_sequence)
    
    if same_sequence:
        file_name = f"{chromosome_name}_DEL{ratios[0]:.2f}_DUP{ratios[1]:.2f}_INV{ratios[2]:.2f}_INS{ratios[3]:.2f}_homo.csv"
    else:
        file_name = f"{chromosome_name}_DEL{ratios[0]:.2f}_DUP{ratios[1]:.2f}_INV{ratios[2]:.2f}_INS{ratios[3]:.2f}_etero.csv"
    
    dataset_path = _parent_path.joinpath(f'dataset/{file_name}')
    df.to_csv(dataset_path, sep=";", index=False)

    create_tsv_dataset(dataset_path)
