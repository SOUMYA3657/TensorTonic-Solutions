import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int) -> np.ndarray:
    B, H, W, C = image.shape
    P = patch_size
    
    # Number of patches along each dimension
    nH = H // P
    nW = W // P
    N = nH * nW
    patch_dim = P * P * C  # flattened patch size
    
    # Reshape: (B, H, W, C) → (B, nH, P, nW, P, C)
    x = image.reshape(B, nH, P, nW, P, C)
    
    # Transpose: → (B, nH, nW, P, P, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    
    # Flatten patches: → (B, N, patch_dim)
    x = x.reshape(B, N, patch_dim)
    
    # Linear projection: (B, N, patch_dim) → (B, N, embed_dim)
    np.random.seed(42)
    W_proj = np.random.randn(patch_dim, embed_dim) * 0.02
    
    return x @ W_proj  # (B, N, embed_dim)