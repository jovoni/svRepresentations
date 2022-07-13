
import pandas as pd
import numpy as np
from pathlib import Path
from utils import *
import sys

_parent_path = Path(__file__).parent

SVs = {
    0: DEL,
    1: DUP,
    2: INS,
    3: INV
}

def create_homogenous_dataset(N: int, ratios: np.array, chromosome_name: str, same_sequence: bool) -> pd.DataFrame:
    """
    Create a dataset from the same chromosome.
    N : length dataset
    ratios : ratios of SVs to apply, in alphabetical order, e.g DEL, DUP, INS, INV
    chromosome_name : name of the chromosome
    same_sequence : will perform SVs on the same sequence
    """

    # Normalize ratios
    ratios = ratios / np.sum(ratios)

    # Create name of file 
    file_name = _parent_path.joinpath(f"data/chromosomes/{chromosome_name}.fa")

    # Initialize columns
    if same_sequence:
        seq = extract_sequence(file_name=file_name, N=1)
        original_sequences = np.full(N, seq[0])
    else:
        original_sequences = extract_sequence(file_name=file_name, N=N)
    
    # How many auxiliary sequence do we need?
    N_aux = int(ratios[2] * N)
    auxiliary_sequences = extract_sequence(file_name=file_name, N=N_aux)

    breakpoint_1 = np.full(N, None)
    breakpoint_2 = np.full(N, None)
    new_sequences = np.full(N, None)
    chromosome_used = np.full(N, chromosome_name)

    # Prepare SVs to perform
    sv_type = np.random.choice([0,1,2,3], size=N, p=ratios)

    # Produce rows of dataframe
    for i in range(N):
        seq = str(original_sequences[i])
        if sv_type[i] == 2: # if it's an insertion
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
    ratios = np.array(sys.argv[2:6], dtype="float64")
    ratios = ratios / np.sum(ratios)
    chromosome_name = sys.argv[6]
    same_sequence = eval(sys.argv[7])

    df = create_homogenous_dataset(N=N, ratios=ratios, chromosome_name=chromosome_name, same_sequence=same_sequence)
    
    if same_sequence:
        file_name = f"{chromosome_name}_DEL{ratios[0]:.2f}_DUP{ratios[1]:.2f}_INV{ratios[2]:.2f}_INS{ratios[3]:.2f}_homo.csv"
    else:
        file_name = f"{chromosome_name}_DEL{ratios[0]:.2f}_DUP{ratios[1]:.2f}_INV{ratios[2]:.2f}_INS{ratios[3]:.2f}_etero.csv"
    
    df.to_csv(_parent_path.joinpath(f'dataset/{file_name}'), sep=";", index=False)