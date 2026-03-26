import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    # Convert to numpy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    # Handle 1D case → reshape to (n,1)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    
    # Compute pairwise distances using broadcasting
    diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))  # shape (n_test, n_train)
    
    # Sort distances and get indices
    sorted_indices = np.argsort(distances, axis=1)
    
    # Handle k > n_train
    if k > n_train:
        result = -1 * np.ones((n_test, k), dtype=int)
        result[:, :n_train] = sorted_indices[:, :n_train]
    else:
        result = sorted_indices[:, :k]
    
    return result