import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    # Convert inputs to NumPy arrays
    p = np.array(p)
    y = np.array(y)

    p = np.clip(p, 1e-15, 1 - 1e-15)

    # term1: for y = 1
    term1 = (1 - p) ** gamma * y * np.log(p)

    # term2: for y = 0
    term2 = p ** gamma * (1 - y) * np.log(1 - p)

    # final loss
    loss = -(term1 + term2)

    return np.mean(loss)