import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    points = np.asarray(points)

    # Check if single point
    single_point = False
    if points.ndim == 1:
        points = points.reshape(1, 3)
        single_point = True

    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack((points, ones))  # Shape: (N, 4)

    # Apply transformation
    transformed_h = (T @ points_h.T).T  # Shape: (N, 4)

    # Convert back to 3D coordinates
    transformed = transformed_h[:, :3]

    # Return original shape
    if single_point:
        return transformed.reshape(3,)
    return transformed