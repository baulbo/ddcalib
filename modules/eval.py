import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def plot_projections(
    depth_maps: list[np.ndarray],
    background_img: np.ndarray,
    width: int,
    height: int,
    out_paths: list[str] = [],
    show_plots: bool = False,
    scatter: list[bool] = None,
) -> None:
    """Plots depth maps separately on background image.

    Args:
        depth_maps: depth maps to be plotted.
        background_img: image to use as background in plot.
        width: width of each depth_map and the background_img.
        height: height of each depth_map and the background_img.
        out_paths: writes the created plots for each depth map to the output path at the corresponding index.
        show_plots: will plot plots if True.
        scatter: will scatter instead of rasterize if True.
    """
    n = len(depth_maps)
    plt.figure(figsize=(12, 5 * n), dpi=96, tight_layout=True)

    for i, (depth_map, out_path) in enumerate(zip(depth_maps, out_paths)):
        plt.subplot(n, 1, i + 1)
        plt.axis([0, width, height, 0])
        plt.imshow(background_img)

        if scatter is None or scatter[i] == False:
            plt.imshow(depth_map, cmap="rainbow_r", vmin=0, vmax=23, alpha=1)
        else:
            ys, xs = np.nonzero(depth_map)
            zs = depth_map[ys, xs]
            plt.scatter(
                xs, ys, c=zs, cmap="rainbow_r", vmin=0, vmax=23, s=0.3, marker="s"
            )

        plt.xticks([])  # removes ticks on x-axis
        # plt.title(out_path)

    # Save entire stacked plot (optional)
    if any(out_paths):
        plt.savefig("stacked_depth_maps.png", bbox_inches="tight")

    if show_plots:
        plt.show()

    plt.close()


def compute_calibration_errors(
    T_pred: np.ndarray, T_gt: np.ndarray, decimals=3, log=True, only_rot_magnitude=False
):
    """Computes the calibration errors between the two given rigid transformations.

    Returns:
        - rotation error magnitude, roll_err, pitch_err, yaw_err in degrees
        - translation error, tx_err, ty_err, tz_err magnitude in cm.
    """
    # compute E_R
    r_err_magnitude, roll_err, pitch_err, yaw_err = compute_relative_rotation_error(
        T_pred[:3, :3], T_gt[:3, :3], only_magnitude=only_rot_magnitude
    )
    if only_rot_magnitude:
        return r_err_magnitude

    # compute E_t
    t_diff = T_gt[:3, 3] - T_pred[:3, 3]
    t_err = np.linalg.norm(t_diff[:, np.newaxis], axis=1)
    t_err_cm = t_err * 100
    t_err_magnitude = np.linalg.norm(t_err_cm[:])

    # round errors to given number of decimals
    (
        r_err_magnitude,
        roll_err,
        pitch_err,
        yaw_err,
        t_err_magnitude,
        tx_err,
        ty_err,
        tz_err,
    ) = np.round(
        np.array(
            [r_err_magnitude, roll_err, pitch_err, yaw_err, t_err_magnitude, *t_err_cm]
        ),
        decimals=decimals,
    )
    if log:
        print("Rotation errors:")
        print(f"E_R = {r_err_magnitude}")
        print(f"Roll = {roll_err}, Pitch = {pitch_err}, Yaw = {yaw_err}")
        print("-----------------------------------------------------")
        print("Translation errors:")
        print(f"E_t {t_err_magnitude} (in cm)")
        print(f"tx = {tx_err}\nty = {ty_err}\ntz = {tz_err}")

    return (
        r_err_magnitude,
        roll_err,
        pitch_err,
        yaw_err,
        t_err_magnitude,
        tx_err,
        ty_err,
        tz_err,
    )


def compute_relative_rotation_error(
    R1_mat: np.ndarray, R2_mat: np.ndarray, only_magnitude: bool = False
) -> tuple:
    """
    Computes the yaw, pitch, and roll error between two rotation matrices
        according to x-right, y-down, z-forward camera frame convention (ZXY Euler angles).

    Reference:
        - http://www.boris-belousov.net/2016/12/01/quat-dist/

    Args:
        R1_mat: first 3x3 rotation matrix.
        R2_mat: second 3x3 rotation matrix.

    Returns:
        (magnitude, roll, pitch, yaw) absolute errors in degrees.
    """
    # relative rotation error
    R_err = R1_mat.T @ R2_mat
    trace = np.trace(R_err)
    magnitude_rad = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
    magnitude = np.degrees(magnitude_rad)

    if only_magnitude:
        return magnitude, None, None, None

    # extract Euler angles
    R_err = Rotation.from_matrix(R_err)
    roll_err, yaw_err, pitch_err = R_err.as_euler("zxy", degrees=True)

    # return all rotation errors in degrees
    return (
        magnitude,
        abs(roll_err),  # around Z
        abs(pitch_err),  # around X
        abs(yaw_err),  # around Y
    )
