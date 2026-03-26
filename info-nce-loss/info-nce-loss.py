import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    # Convert to numpy arrays
    Z1 = np.array(Z1)
    Z2 = np.array(Z2)
    # Compute similarity matrix
    S = np.dot(Z1, Z2.T) / temperature  # (N, N)
    # Numerical stability: subtract row-wise max
    S_max = np.max(S, axis=1, keepdims=True)
    S_stable = S - S_max
    # Compute exp
    exp_S = np.exp(S_stable)
    # Compute log-softmax denominator
    log_sum_exp = np.log(np.sum(exp_S, axis=1))
    # Extract diagonal (positive pairs)
    diag = np.diag(S_stable)
    # Compute loss per sample
    loss = - (diag - log_sum_exp)
    # Return mean loss
    return np.mean(loss)