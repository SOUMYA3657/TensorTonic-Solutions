import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)
    y: array of shape (N,) with values in {0,1}
    """
    # Convert to numpy arrays
    a = np.array(a)
    b = np.array(b)
    y = np.array(y)
    
    # Handle single vector case → reshape to (1, D)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    
    # Ensure y is (N,)
    y = y.reshape(-1)
    
    # Validate y values
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only 0 or 1")
    
    # Compute Euclidean distance
    d = np.linalg.norm(a - b, axis=1)
    
    # Compute loss
    loss = y * (d ** 2) + (1 - y) * (np.maximum(0, margin - d) ** 2)
    
    # Reduction
    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")