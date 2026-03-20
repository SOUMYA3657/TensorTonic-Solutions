import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    N, D = X.shape
    w = np.zeros(D)
    b = 0.0

    for _ in range(steps):
        p = _sigmoid(X @ w + b)
        error = p - y
        grad_w = (X.T @ error) / N
        grad_b = np.mean(error)
        w -= lr * grad_w
        b -= lr * grad_b

    return (w, b)