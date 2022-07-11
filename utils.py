
import random 
import re
import numpy as np

MAX_LENGTH = 510
MIN_SV_LENGTH = 50

def INV(seq):
    """
    Reverse a portion of the sequence.
    IN  : sequence
    OUT : modified sequence, breakpoint 1, breakpoint 2, SV length
    """
    assert_sequence_validity(seq)
    
    # Choose breakpoints
    l = len(seq)
    inv_length = random.randint(MIN_SV_LENGTH, l)
    b = random.randint(0, l - inv_length) # breakpoint 1     
    s, e = b, b + inv_length # breakpoint 1 and 2
    
    # Modify sequence
    seq_start = seq[:s]
    seq_end = seq[e:]
    portion_to_invert = seq[s:e]
    inverted_portion = portion_to_invert[::-1]
    new_seq = seq_start + inverted_portion + seq_end 
    
    return new_seq, s, e, inv_length

def DEL(seq):
    """
    Reverse a portion of the sequence.
    IN  : sequence
    OUT : modified sequence, breakpoint 1, breakpoint 2, SV length
    """
    assert_sequence_validity(seq)
    
    # Choose breakpoints
    l = len(seq)
    del_length = random.randint(MIN_SV_LENGTH, l) 
    b = random.randint(0, l - del_length) # compute breakpoint
    s, e = b, b + del_length # breakpoint 1 and 2
    
    # Modify sequence
    seq_start = seq[:s]
    seq_end = seq[e:]
    new_seq = seq_start + seq_end 
    
    return new_seq, s, e, del_length 

def DUP(seq):
    """
    Duplicate a portion of the sequence.
    IN  : sequence
    OUT : modified sequence, breakpoint 1, breakpoint 2, SV length
    """
    assert_sequence_validity(seq)
    
    # Choose breakpoints
    l = len(seq)
    dup_length = random.randint(MIN_SV_LENGTH, l)
    b = random.randint(0, l - dup_length)
    s, e = b, b + dup_length
    
    # Modify sequence
    seq_start = seq[:s]
    seq_end = seq[e:]
    portion_to_duplicate = seq[s:e]
    new_seq = seq_start + portion_to_duplicate + portion_to_duplicate + seq_end 
    
    # Return modified sequence, with breakpoint 1 and 2
    return new_seq, s, e, dup_length

def INS(seq_1, seq_2):
    """
    Insert into first sequence a portion of the second sequence.
    IN  : sequence1, sequence2
    OUT : modified sequence, breakpoint 1, breakpoint 2, SV length
    """
    assert_sequence_validity(seq_1)
    assert_sequence_validity(seq_2)
    
    # Choose breakpoints
    l_1 = len(seq_1)
    l_2 = len(seq_2)
    max_ins_length = min(MAX_LENGTH - l_1, l_2)
    assert max_ins_length >= MIN_SV_LENGTH, "Trying to perform insertion smaller than 50!"
    ins_length = random.randint(MIN_SV_LENGTH, max_ins_length)
    b1 = random.randint(0, l_1)
    b2 = random.randint(0, l_2 - ins_length)
    
    # Modify sequence
    seq_start = seq_1[:b1]
    seq_end = seq_1[b1:]
    portion_to_insert = seq_2[b2:b2 + ins_length]
    new_seq = seq_start + portion_to_insert + seq_end
    
    return new_seq, b1, b1 + ins_length, ins_length

def extract_sequence(file_name, N):
    with open(file_name, 'r') as f:
        _ = f.readlines(1)[0]
        text = f.read()
    
    text = re.sub('([N,n,\n])', "", text)
    text_l = len(text)

    sequences = np.full(N, None)

    for i in range(N):
        seq_l = random.randint(MIN_SV_LENGTH + 1, MAX_LENGTH - MIN_SV_LENGTH)
        new_seq_start = random.randint(0, text_l - seq_l + 1)
        new_seq = text[new_seq_start:new_seq_start + seq_l]
        sequences[i] = new_seq

    return sequences

def assert_sequence_validity(seq):
    assert isinstance(seq, str), f"Passed argument should be a string!"
    assert len(seq) <= MAX_LENGTH, f"Sequence is longer than maximum lenght! {len(seq)} > {MAX_LENGTH}"
    assert len(seq) > 50, f"Sequence is not long enough! 50 bp is minimum!"