import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    if not seqs:
        return np.empty((0, 0), dtype=int)
    
    if max_len is None:
        max_len = max(len(s) for s in seqs)
    
    result = np.full((len(seqs), max_len), pad_value, dtype=int)
    
    for i, seq in enumerate(seqs):
        trunc = seq[:max_len]
        result[i, :len(trunc)] = trunc
    
    return result