import numpy as np
import os
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt


def confidence_mask(uncertainty_map, conf_th=50):
    """Masks out given percentage of the points with lowest confidence
    (or highest uncertainty, i.e the estimated scale-invaraint log error)."""
    # Convert percentage threshold to actual confidence value
    # Source: https://github.com/facebookresearch/vggt
    confidence_map = 1 / (
        uncertainty_map + 1e-6
    )  # NOTE: confidences are normally not 0, but nonetheless..
    if conf_th == 0.0:
        conf_thhold = 0.0
    else:
        conf_thhold = np.percentile(confidence_map, conf_th)

    conf_mask = (confidence_map >= conf_thhold) & (confidence_map > 1e-5)
    return conf_mask


def fibonacci_sphere(num_points: int, show_plot: bool = False):
    """Creates equidistant points on the surface of a sphere using Fibonacci sphere algorithm.

    GitHub source (not my implementation):
        - https://gist.github.com/Seanmatthews/a51ac697db1a4f58a6bca7996d75f68c
        - by Seanmatthews
    """
    ga = (3 - np.sqrt(5)) * np.pi  # golden angle

    # Create a list of golden angle increments along tha range of number of points
    theta = ga * np.arange(num_points)

    # Z is a split into a range of -1 to 1 in order to create a unit circle
    z = np.linspace(1 / num_points - 1, 1 - 1 / num_points, num_points)

    # a list of the radii at each height step of the unit circle
    radius = np.sqrt(1 - z * z)

    # Determine where xy fall on the sphere, given the azimuthal and polar angles
    y = radius * np.sin(theta)
    x = radius * np.cos(theta)

    # Display points in a scatter plot
    if show_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z)
        plt.show()
        plt.close()
    return np.vstack((x, y, z)).T


def get_error_transform(direction: np.ndarray, R_error=0.0, t_error=0.0):
    """Constructs a rigid transform with given rotation and translation magnitudes applied along the direction.

    Args:
        direction: a unit vector of shape (3,) in direction to apply the errors.
        R_error: exact magnitude of rotation error to apply in degrees.
        t_error: exact magnitude of translation error to apply in meters.

    Returns:
        Transformation with the given errors applied with respect to given unit vector.
    """
    # apply rotation error
    R = np.eye(3)
    if R_error != 0.0:
        rad = np.deg2rad(R_error)
        R = Rotation.from_rotvec(direction * rad)
        R = R.as_matrix()

    # add translation error
    if t_error > 0:
        # scale by given magnitude
        error_t = direction * t_error
    else:
        error_t = np.zeros(3)

    error_T = np.eye(4)
    error_T[:3, :3] = R
    error_T[:3, 3] = error_t
    return error_T


def homogenization_by_minimal_pruning(points: list[np.ndarray]) -> np.ndarray:
    """Limits number of points per point cloud to minimal number (=N_min) of points
            and makes points homogeneous.

    Args:
        points: a list of point clouds of (inhomogeneous) shape (N_i, 3).

    Returns:
        points as homogeneous np.ndarray of shape (B, 4, N_min).
    """
    min_n = points[0].shape[0]
    if len(points) > 1:
        min_n = min([p.shape[0] for p in points])
    pruned_points = [make_homogeneous(p[:min_n, :]) for p in points]
    return np.array(pruned_points)


def make_homogeneous(points: np.ndarray) -> np.ndarray:
    """Converts point cloud of shape (N, 3) to homogeneous array of shape (4 x N).

    Args
        points: point cloud of shape (N, 3)

    Returns: point clouds as homogeneous np.ndarray of shape (4, N).
    """
    points = np.concatenate(
        (points.T, np.ones(shape=(1, points.T.shape[1]), dtype=np.float32)), axis=0
    )
    return points
