import numpy as np
import os
import matplotlib.image as mpimg

from modules.preprocessing import confidence_mask

def load_images(frame_paths: list[str]) -> np.ndarray:
    """Loads images of homogeneous shape at given paths as numpy array.

    Args:
        frame_paths: paths to the frames the frames that should be loaded as homogeneous numpy array.

    Returns:
        images as homogeneous numpy array of shape (B, 3, h, w).
    """
    image_i = mpimg.imread(frame_paths[0])
    images = np.zeros((len(frame_paths), *image_i.shape))
    images[0, :] = image_i
    i = 1
    for frame_path in frame_paths[1:]:  # add the other frames to the sequences
        images[i, :] = mpimg.imread(frame_path)
        i += 1
    return images


def load_vggt_frame(
    base_path: str, frame_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads depth, mask, and image frame saved to directory at `base_path` by VGGT preprocessing.

    Args:
        base_path: the path to directory containing `base_path`+`frame_names[i]`+{.png, _depth.npy, _mask.npy}
        frame_name: the name of the frame (incl. leading zeros) without file extension.

    Returns:
        (image, depth map, mask) of shapes ( (h, w, 3), (h, w), (h, w) ).
    """
    # read VGGT's prediction output (i.e mask, depth prediction, and the cropped image)
    vggt_depth_path = os.path.join(base_path, str(frame_name) + "_depth.npy")
    mask_path = os.path.join(base_path, str(frame_name) + "_mask.npy")
    image_path = os.path.join(base_path, str(frame_name) + ".png")

    # read VGGT depth map, and mask
    vggt_depth = np.load(vggt_depth_path).squeeze()
    vggt_mask = np.load(mask_path)
    vggt_image = mpimg.imread(image_path)
    return (vggt_image, vggt_depth, vggt_mask.astype(bool))


def load_unidepth_output(
    base_path: str, frame_names: list[str], conf_th: float = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Loads UniDepthV2 models predictions of confidence and depth maps.

    Args:
        base_path: the path to directory containing `base_path`+`frame_names[i]`+{.png, _depth.npy, _uncertainty.npy}
        frame_name: the name of the frame (incl. leading zeros) without file extension.
        conf_th: confidence threshold, i.e percentage 0 - 100 of most confident depth predictions to keep.

    Returns:
        ( depth maps, masks ) of shapes ( (B, h, w), (B, h, w) ).
    """

    unidepth_depth_i = np.load(os.path.join(base_path, frame_names[0] + "_depth.npy"))
    unidepth_uncertainty_i = np.load(
        os.path.join(base_path, frame_names[0] + "_uncertainty.npy")
    )
    unidepth_depth = np.zeros((len(frame_names), *unidepth_depth_i.shape))
    unidepth_mask = np.zeros(
        (len(frame_names), *unidepth_uncertainty_i.shape), dtype=bool
    )
    unidepth_depth[0, :] = unidepth_depth_i
    unidepth_mask[0, :] = confidence_mask(unidepth_uncertainty_i, conf_th)
    i = 1
    for frame_name in frame_names[1:]:
        unidepth_depth_i = np.load(os.path.join(base_path, frame_name + "_depth.npy"))
        unidepth_uncertainty_i = np.load(
            os.path.join(base_path, frame_name + "_uncertainty.npy")
        )
        unidepth_depth[i, :] = unidepth_depth_i
        unidepth_mask[i, :] = confidence_mask(unidepth_uncertainty_i, conf_th)
        i += 1
    return unidepth_depth, unidepth_mask


def load_vggt_output(
    base_path: str, frame_names: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reads VGGT's rescaled input images and depth maps with masks.

    Args:
        base_path: the path to directory containing `base_path`+`frame_names[i]`+{.png, _depth.npy, _mask.npy}.
        frame_names: list of the name of the frames (incl. leading zeros) without file extension.

    Returns:
        ( images, depth maps, masks ) of shapes ( (B, h, w, 3), (B, h, w), (B, h, w) ).
    """
    vggt_image_i, vggt_depth_i, vggt_mask_i = load_vggt_frame(base_path, frame_names[0])
    vggt_depth = np.zeros((len(frame_names), *vggt_depth_i.shape))
    vggt_depth[0, :] = vggt_depth_i
    vggt_mask = np.zeros((len(frame_names), *vggt_mask_i.shape), dtype=bool)
    vggt_mask[0, :] = vggt_mask_i
    vggt_image = np.zeros((len(frame_names), *vggt_image_i.shape))
    vggt_image[0, :] = vggt_image_i
    i = 1
    for frame_name in frame_names[1:]:  # add the other frames to the sequences
        # add next VGGT estimates
        vggt_image_i, vggt_depth_i, vggt_mask_i = load_vggt_frame(base_path, frame_name)
        vggt_depth[i, :] = vggt_depth_i
        vggt_mask[i, :] = vggt_mask_i
        vggt_image[i, :] = vggt_image_i
        i += 1
    return (vggt_image, vggt_depth, vggt_mask)


def load_depthpro_output(base_path: str, frame_names: list[str]):
    """
    Args:
        base_path: the path to directory containing `base_path`+`frame_names[i]`+{.png, _depth.npy, _mask.npy}
        frame_name: the name of the frame (incl. leading zeros) without file extension.
    """
    depthpro_depth_i = np.load(os.path.join(base_path, frame_names[0] + ".npy"))
    depthpro_depth = np.zeros((len(frame_names), *depthpro_depth_i.shape))
    depthpro_depth[0, :] = depthpro_depth_i
    i = 1
    for frame_name in frame_names[1:]:
        depthpro_depth_i = np.load(os.path.join(base_path, frame_name + ".npy"))
        depthpro_depth[i, :] = depthpro_depth_i
        i += 1
    return depthpro_depth


def load_kitti_cam_calibration(
    cam_to_cam_path: str, camera_id: int, scaling: tuple[float] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Loads the P_rect_0{`camera_id`} and R_rect_0{`camera_id`} matrices from KITTI's
    calib_cam_to_cam.txt file.

    Ref:
        - https://github.com/pratikac/kitti/blob/master/readme.raw.txt

    Args:
        cam_to_cam_path: path to the calib_cam_to_cam.txt file of the used KITTI sequence.
        camera_id: index of the camera for which projection and rectification matrices should be loaded.
        scaling: scaling factors to be applied to camera intrinsics (i.e s*fx, s*fy, s*cx, s*cy) as ( scale_x, scale_y ).

    Returns:
        (scaled) projection matrix (after rect.) P and rectifying rotation R_rect making image plane co-planar.
    """
    # read projection matrix from KITTI cam_to_cam calibration file
    with open(cam_to_cam_path, "r") as f:
        l = f.readline()
        tokens = l.strip("\n").split(" ")

        # NOTE: assumes R_rect comes before P_rect
        R_rect = None
        while l and tokens[0][:-1] != "R_rect_0" + str(0):
            l = f.readline()
            tokens = l.strip("\n").split(" ")

        R_rect = np.array([float(x) for x in tokens[1:]]).reshape(3, 3)
        R_rect = np.insert(R_rect, 3, values=[0, 0, 0], axis=0)
        R_rect = np.insert(R_rect, 3, values=[0, 0, 0, 1], axis=1)

        P_rect = None
        while l and tokens[0][:-1] != "P_rect_0" + str(camera_id):
            l = f.readline()
            tokens = l.strip("\n").split(" ")

        P_rect = np.array([float(x) for x in tokens[1:]]).reshape(3, 4)
        if scaling:  # apply scaling to fu, fv, cx, cy to make up for output dimensions
            P_rect[0][0], P_rect[0][2] = (
                P_rect[0][0] * scaling[0],
                P_rect[0][2] * scaling[0],
            )  # scale_x
            P_rect[1][1], P_rect[1][2] = (
                P_rect[1][1] * scaling[1],
                P_rect[1][2] * scaling[1],
            )  # scale_y
        return P_rect @ R_rect


def load_kitti_velo_to_cam_calibration(file_path: str) -> np.ndarray:
    """Loads extrinsic LiDAR to cam calibration matrix from KITTI formatted file.

    Args:
        file_path: path to the calib_velo_to_cam.txt calibration file.

    Notes:
        Expected file format is
        '''
            calib_time: 15-Mar-2012 11:37:16
            R: 7.533745e-03 -9.999714e-01 -6.166020e-04 1.480249e-02 7.280733e-04 -9.998902e-01 9.998621e-01 7.523790e-03 1.480755e-02
            T: -4.069766e-03 -7.631618e-02 -2.717806e-01
            delta_f: 0.000000e+00 0.000000e+00
            delta_c: 0.000000e+00 0.000000e+00
        '''

    Returns: 4x4 extrinsic calibration matrix.
    """
    T_calib = np.zeros((3, 4), dtype=np.float32)
    with open(file_path, "r") as calib_file:
        lines = calib_file.readlines()
        R = np.array(lines[1].split()[1:], dtype=np.float32)  # extract rotation (R:)
        T_calib[0:3, 0:3] = np.reshape(R, (3, 3))
        T_calib[:, 3] = np.array(
            lines[2].split()[1:], dtype=np.float32
        )  # extract rotation (R:)
    T_calib = np.concatenate(
        (T_calib, np.array([[0, 0, 0, 1]], dtype=np.float32)), dtype=np.float32
    )
    return T_calib
