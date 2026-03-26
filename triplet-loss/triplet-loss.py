import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    # Convert to numpy arrays
    anchor = np.array(anchor)
    positive = np.array(positive)
    negative = np.array(negative)
    
    # Handle 1D input → reshape to (1, D)
    if anchor.ndim == 1:
        anchor = anchor.reshape(1, -1)
    if positive.ndim == 1:
        positive = positive.reshape(1, -1)
    if negative.ndim == 1:
        negative = negative.reshape(1, -1)
    
    # Compute squared Euclidean distances
    d_ap = np.sum((anchor - positive) ** 2, axis=1)
    d_an = np.sum((anchor - negative) ** 2, axis=1)
    
    # Compute triplet loss
    loss = np.maximum(0, d_ap - d_an + margin)
    
    # Return mean loss
    return np.mean(loss)