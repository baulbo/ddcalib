from scipy.spatial.transform import Rotation
import numpy as np
import os
import quaternion  # NOTE: use this instead of scipy bec. it has built in normalization of quat
import matplotlib.image as mpimg

from modules.io import load_kitti_velo_to_cam_calibration

NUM_BINS = 128

def parameterize_transformation(
    T: np.ndarray, rotation_only: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Parameterizes rigid transform `T` as parameter vector.

    Args:
        T: a 4x4 or 3x3 [R, t] rigid transformation matrix.
        rotation_only: if True, only rotation is parameterized as p.

    Returns:
        tuple:
            - If rotation_only is False: (p=[qw, qx, qy, qz, tx, ty, tz], t = None, Rp = None)
            - If rotation_only is True: (p=[qw, qx, qy, qz], t = [tx, ty, tz], Rp = None)
    """
    t = T[0:3, 3].flatten()
    q = quaternion.from_rotation_matrix(T[0:3, 0:3])
    q = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
    if rotation_only:
        return q, t, None  # optimize R, fixed t
    p = np.concatenate([q, t], axis=0)  # FIXME: this is not used in the final method...
    return p, None, None  # joint optimization, no fixed values


def parameterization_to_transform(
    p: np.ndarray, R_mat: np.ndarray = None
) -> np.ndarray:
    """Generates 4x4 rigid transformation matrix from [qw, qx, qy, qz, tx, ty, tz] parameterization.

    Args:
        p: parameter vector [qw, qx, qy, qz, tx, ty, tz].
        R_mat: 3x3 matrix form of rotation if it is to be kept fixed.

    Returns:
        a 4x4 rigid transformation matrix.
    """
    T = np.eye(4)
    # rotation
    if R_mat is not None:
        T[0:3, 0:3] = R_mat
        T[0:3, 3] = p
    else:
        T[0:3, 0:3] = quaternion.as_rotation_matrix(
            np.quaternion(p[0], p[1], p[2], p[3])
        )
        T[0:3, 3] = p[4:]
    return T


def project_points(P: np.ndarray, points: np.ndarray, dims: tuple):
    """Projects given points in camera coordinate frame onto (rasterized)
        virtual image using P keeping closest depth (z) value.

    Based on this implementation <https://github.com/SanghyunPark01/KITTI_L2C_Projection_ROS>
        by SanghyunPark01.

    Args:
        P: 3x4 projection matrix.
        points: batch of homogeneous points [x, y, z, w]^T in camera coords of shape (B, 4, N).

    Returns:
        a depth map of shape (dims[0] x dims[1]).
    """
    points_T = points.transpose(
        0, 2, 1
    )  # gives (B, N, 4) to prevent weird boolean indexing axes swaps
    rasterized_proj_batch = np.zeros((points.shape[0], *dims), dtype=np.float32)
    for bi in range(points_T.shape[0]):
        # filter points not in front of camera (i.e negative Z)
        valid = points_T[bi, :, 2] > 0
        points_i = points_T[bi, valid, :]

        # project points
        proj = P @ points_i.T

        # dehom
        proj[:2] /= proj[2, :]

        # filter point out of canvas
        img_h, img_w = dims[0], dims[1]
        u, v, z = proj
        u_out = np.logical_or(u < 0, u >= img_w)
        v_out = np.logical_or(v < 0, v >= img_h)
        outlier = np.logical_or(u_out, v_out)
        proj = np.delete(proj, np.where(outlier), axis=1)

        # generate color map from depth (u = width, v = height)
        u, v, z = proj

        # rasterize
        u, v = np.floor(u).astype(int), np.floor(v).astype(int)
        # keep only the closest z
        lidar_projection_image = np.full((img_h * img_w), np.inf)
        flat_indices = v * img_w + u
        np.minimum.at(lidar_projection_image, flat_indices, z)
        lidar_projection_image = lidar_projection_image.reshape((img_h, img_w))
        lidar_projection_image[
            lidar_projection_image == np.inf
        ] = 0  # replace no depth (inf) by zero
        rasterized_proj_batch[bi] = lidar_projection_image
    return rasterized_proj_batch


def cross_modal_min_max_normalize(x1: np.ndarray, x2: np.ndarray, mask: np.ndarray):
    """Cross-modal min max normalization.

    Args:
        x1: samples from the first modality of shape (B, N).
        x2: samples from the second modality of shape (B, N).
        mask: binary mask of shape (B, N).

    Returns:
        min max normalized x based on masked x elements.
    """
    masked_x1 = np.where(mask, x1, np.nan)
    masked_x2 = np.where(mask, x2, np.nan)

    masked_xs = np.concatenate([masked_x1, masked_x2], axis=1)

    z_min = np.nanmin(masked_xs, axis=1, keepdims=True)  # gives shape (B, 1)
    z_max = np.nanmax(masked_xs, axis=1, keepdims=True)

    z_diff = z_max - z_min
    z_diff[z_diff == 0] = 1

    normalized_x1, normalized_x2 = np.clip((x1 - z_min) / z_diff, 0, 1), np.clip(
        (x2 - z_min) / z_diff, 0, 1
    )
    normalized_x1[~mask], normalized_x2[~mask] = (
        0,
        0,
    )  # FIXME: setting this to zero has no meaning (and is incorrect) given the mask (it works bec. we use the mask)

    return normalized_x1, normalized_x2


def min_max_normalize(x: np.ndarray, mask: np.ndarray):
    """Min max normalization.

    Args:
        x: samples of shape (B, N).
        masks: binary mask of shape (B, N).

    Returns:
        min max normalized x based on masked x elements.
    """
    masked_x = np.where(mask, x, np.nan)

    z_min = np.nanmin(masked_x, axis=1, keepdims=True)  # gives shape (B, 1)
    z_max = np.nanmax(masked_x, axis=1, keepdims=True)

    z_diff = z_max - z_min
    z_diff[z_diff == 0] = 1

    normalized_x = (x - z_min) / z_diff
    normalized_x[~mask] = 0
    return normalized_x


def mask_depth_map(
    lidar_depth: np.ndarray,
    cam_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Masks the depth maps by applying vggt_mask (i.e confidence and sky mask) and optionally projection mask.

    Args:
        lidar_depth: (sparse) LiDAR depth map.
        cam_mask: camera-related mask (e.g sky mask, confidence masks...)
        projection_mask: the sparsity of the LiDAR depth map is applied as a mask to the VGGT (to get equal amount of samples) when True, not applied otherwise.

    Returns:
        a mask that when used as x[mask] gives valid depth values.
    """
    # remove (i.e set to zero) depth values at pixels without LiDAR projection
    if cam_mask is None:
        full_mask = lidar_depth == 0
    else:
        full_mask = (lidar_depth == 0) | (~cam_mask)
    return ~full_mask


def batched_histogram(
    x1: np.ndarray,
    x2: np.ndarray,
    mask: np.ndarray,
    num_bins=NUM_BINS,
    hist_range: tuple[float, float] = None
):
    """Computes histograms of samples per batch.

    Args:
        x1: samples batch of shape (B, N).
        x2: samples batch of shape (B, N).
        mask: binary mask of shape (B, N). Determines which samples are used to compute histogram.
        num_bins: number of bins to use.
        hist_range: histogram range in meters.

    Returns:
        P1: probability distr. of shape (B, num_bins).
        P2: probability distr. of shape (B, num_bins).
        P_joint: probability distr. of shape (B, num_bins, num_bins).
    """
    B = x1.shape[0]
    P1, P2, P_joint = (
        np.zeros((B, num_bins), dtype=np.float32),
        np.zeros((B, num_bins), dtype=np.float32),
        np.zeros((B, num_bins, num_bins), dtype=np.float32),
    )
    for bi in range(B):
        P1[bi], P2[bi], P_joint[bi] = histogram(
            x1=x1[bi],
            x2=x2[bi],
            mask=mask[bi],
            num_bins=num_bins,
            hist_range=hist_range,
        )
    return P1, P2, P_joint


def histogram(
    x1: np.ndarray,
    x2: np.ndarray,
    mask: np.ndarray,
    num_bins=NUM_BINS,
    hist_range: tuple[float, float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes histograms of normalized samples with bin edges ranging from 0 to 1.

    Args:
        x1: samples shape (N).
        x2: samples shape (N).
        mask: binary mask of shape (N). Determines which samples are used to compute histogram.
        num_bins: number of bins to use.
        hist_range: fixed min and max value of histogram (bin range).

    Returns:
        P1: probability distr. of shape (num_bins).
        P2: probability distr. of shape (num_bins).
        P_joint: probability distr. of shape (num_bins, num_bins).
    """
    P_joint_count, _, _ = np.histogram2d(
        x1[mask],
        x2[mask],
        bins=[num_bins, num_bins],
        range=[hist_range, hist_range],

    )
    P_joint_count_sum = P_joint_count.sum()
    if P_joint_count_sum == 0:
        P_joint = np.zeros((num_bins, num_bins), dtype=float)
    else:
        P_joint = P_joint_count / P_joint_count_sum
    P1 = np.sum(P_joint, axis=1)
    P2 = np.sum(P_joint, axis=0)
    return P1, P2, P_joint


def compute_nid(P1: np.ndarray, P2: np.ndarray, P_joint: np.ndarray) -> float:
    """Computes the Normalized Information Distance between given (density) histograms.

    Args:
        P1: probability distr. of shape (B, num_bins).
        P2: probability distr. of shape (B, num_bins).
        P_joint: probability distr. of shape (B, num_bins, num_bins).

    Returns:
        nid for each batch.
    """
    # compute entropies
    eps = 1e-6
    H1 = -np.sum(P1 * np.log(P1 + eps), axis=1)
    H2 = -np.sum(P2 * np.log(P2 + eps), axis=1)
    H_joint = -np.sum(P_joint * np.log(P_joint + eps), axis=(1, 2))
    mi = H1 + H2 - H_joint
    nid = (H_joint - mi) / H_joint
    return np.clip(nid, 0, 1)


def average_nid(P1: np.ndarray, P2: np.ndarray, P_joint: np.ndarray) -> float:
    """Computes the average Normalized Information Distance along batch dimension.

    Args:
        P1: probability distr. of shape (B, num_bins).
        P2: probability distr. of shape (B, num_bins).
        P_joint: probability distr. of shape (B, num_bins, num_bins).

    Returns:
        average NID.
    """
    nids = compute_nid(P1, P2, P_joint)
    avg_nid = np.mean(nids)
    if np.isnan(avg_nid):
        return 1.0  # return max NID in case of nan
    return avg_nid


def loss_pipeline(
    p: np.ndarray,
    points: np.ndarray,
    P: np.ndarray,
    cam_depth: np.ndarray,
    init_mask: np.ndarray,
    dims: tuple[int, int],
    t: np.ndarray = None,
    R_mat: np.ndarray = None,
    num_bins=NUM_BINS,
) -> float:
    """Computes average NID (i.e the objective to be minimized).

    Args:
        p: parameter vector [*q of shape (4), tx, ty, tz] if t is None, p=q, i.e a quaternion of shape (4,) otherwise.
        points: point clouds of shape (B x 4 x N).
        P: projection matrix of shape (3, 4)
        cam_depth: input (depth map) samples from first camera of shape (B, h, w).
        init_mask: mask that has to be applied to depth maps, also of shape (B, h, w). # e.g confidence / sky mask.
        dims: (height, width) of depth map and masks.
        t: should be passed if translation is to be kept fixed.
        R_mat: should be passed if rotation is to be kept fixed. (as 3x3 matrix R).
        num_bins: number of bins to use for histograms.

    Returns: average NID.
    """
    # construct extrinsic calibration estimate
    if t is not None:
        # t is to be kept fixed
        p = np.concatenate((p, t))
        T_calib = parameterization_to_transform(p=p)
    elif R_mat is not None:
        # R is to be kept fixed
        T_calib = parameterization_to_transform(p=p, R_mat=R_mat)
    else:
        T_calib = parameterization_to_transform(p=p)

    # transform LiDAR points to camera frame
    X_c = T_calib @ points  # result is (B, 4, N)

    # project points into virtual image plane
    # NOTE: this function does not support batched np...
    li_depth = project_points(P=P, points=X_c, dims=dims)

    # apply masks
    mask = mask_depth_map(li_depth, init_mask)

    # flatten per batch
    flat_mask, flat_cam_depth, flat_li_depth = (
        mask.reshape(mask.shape[0], -1),  # i.e the full mask
        cam_depth.reshape(cam_depth.shape[0], -1),  # e.g camera samples
        li_depth.reshape(li_depth.shape[0], -1),  # e.g LiDAR
    )

    # trust the LiDAR to tell us the maximum distance (depthpro)

    # normalize values of within each depth map
    # NOTE: uses raw samples (no normalization)
    flat_norm_cam_depth, flat_norm_li_depth = flat_cam_depth, flat_li_depth

    # compute NID for all frames
    P1, P2, P_joint = batched_histogram(
        x1=flat_norm_cam_depth,
        x2=flat_norm_li_depth,
        mask=flat_mask,
        num_bins=num_bins
    )

    avg_nid = average_nid(P1, P2, P_joint)
    return avg_nid
