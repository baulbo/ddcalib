import numpy as np
import os
import quaternion
from modules.preprocessing import homogenization_by_minimal_pruning

from modules.io import (
    load_unidepth_output,
    load_depthpro_output,
    load_vggt_output,
    load_images,
    load_kitti_cam_calibration,
    load_kitti_velo_to_cam_calibration,
)


class Dataset:  # NOTE: currently just a class to load KITTI data (some code is specific to seq. 00)
    def __init__(self, num_frames: int, start_frame_id: int):
        if num_frames < 2:
            print(
                "[WARNING] Using less than 2 frames does not allow for interpolation."
            )

        # General properties
        self.num_frames = num_frames
        self.start_frame_id = start_frame_id
        # store consecutive frame names (excl. file extension) starting from `start_frame_id`
        # TODO: this convention is assumed to be consistent over datasets, should not be the case
        self.frame_ids = [
            f"{id:010d}"
            for id in range(start_frame_id, start_frame_id + self.num_frames)
        ]

        # LiDAR properties
        self.points = None

        # Camera properties
        self.camera_images = None
        self.camera_depths = None
        self.camera_masks = None

        # Calibration properties
        self.P = None  # projection matrix
        self.T_gt = None  # ground truth calibration matrix

    def load_lidar_data(
        self,
        lidar_path: str,
    ):
        """Loads LiDAR point clouds (.bin).

        Args:
            lidar_path: path to directory containing `timestamps.txt` file and a subdirectory named `data`
                containing LiDAR point clouds in KITTI (.bin) format.

        Returns:
            A list of (inhomogeneous) point clouds each of shape shape (N_i, 3).
        """
        #  lidar frames
        #  remove leading zeros for odometry dataset (FIXME: might not generalize to other raw and odometry seq. than 00)
        sliced_frame_ids = [
            frame_id[4:] for frame_id in self.frame_ids
        ]  # NOTE: bec. 001250.bin different amount of leading zeros...

        self.points = [
            np.fromfile(
                os.path.join(lidar_path, "data", str(frame_id) + ".bin"),
                dtype=np.float32,
            ).reshape((-1, 4))[
                :, 0:3
            ]  # N x 3
            for frame_id in sliced_frame_ids
        ]

    def load_camera_data(self, base_path: str, model: str, conf_th: float = 95):
        """Loads camera-based depth estimates, corresponding model input images, and masks.

        Args:
            base_path: path to the directory containing all that is to be loaded from the depth estimation output.
            model: name of the model to use, can be either "VGGT" or "DEPTHPRO" or "UNIDEPTH"
            conf_th: / see load_unidepth_output func.
        """
        if model == "VGGT":
            (
                self.camera_images,
                self.camera_depths,
                self.camera_masks,
            ) = load_vggt_output(base_path=base_path, frame_names=self.frame_ids)
        elif model == "DEPTHPRO":
            self.camera_depths = load_depthpro_output(
                base_path=base_path, frame_names=self.frame_ids
            )
        elif model == "UNIDEPTH":
            (
                self.camera_depths,
                self.camera_masks,
            ) = load_unidepth_output(
                base_path=base_path, frame_names=self.frame_ids, conf_th=conf_th
            )
        else:
            raise Exception(f"The model {model} is not supported.")

        self.image_height, self.image_width = (
            self.camera_depths[0].shape[0],
            self.camera_depths[0].shape[1],
        )

    def load_calibration(
        self,
        images_dir: str,
        camera_calibration_path: str,
        extrinsic_calibration_path: str,
        camera_id: int = 2,  # left RGB camera by default
        image_file_extension: str = ".png",
    ):
        """Sets the camera and extrinsic calibration.

        The scaling factor is determined based on the camera data (i.e data after depth estimation)
            and the images from the original dataset.

        Args:
            images_dir: path to the directory containing the images from the original dataset (i.e before depth estimation preprocessing).
        """
        if self.image_width is None or self.image_height is None:
            raise Exception(
                "The image width and height used for the LiDAR projection should be "
                "known before loading the calibration data in order to apply appropraite scaling."
            )

        # Load single original image to compute image scaling done by monocular depth estimation method used
        original_images = load_images(
            frame_paths=[
                os.path.join(images_dir, self.frame_ids[0] + image_file_extension)
            ]
        )
        original_h, original_w, _ = original_images[0].shape

        # Load KITTI's camera calibration (i.e projection matrix)
        self.P = load_kitti_cam_calibration(
            cam_to_cam_path=camera_calibration_path,
            camera_id=camera_id,
            scaling=(self.image_width / original_w, self.image_height / original_h),
        )
        self.T_gt = load_kitti_velo_to_cam_calibration(extrinsic_calibration_path)

    def preprocess_points(self, interpolate: bool = False, **kwargs):
        """Preprocesses the LiDAR point clouds (i.e points) as a list with np.ndarrays of shape (N_i, 3)
                and makes them np.ndarray of shape (B, 4, N_min).

        Args:
            interpolate: if True, LiDAR points will be interpolated based on timestamps given as kwargs.
        """
        if self.points is None:
            raise Exception("Load LiDAR data before attempting interpolation.")

        if interpolate:
            self._interpolate_points(
                lidar_ts_path=kwargs["lidar_ts_path"],
                images_ts_path=kwargs["images_ts_path"],
                poses_path=kwargs["poses_path"],
                num_frames=kwargs["num_frames_after_interpolation"],
            )

        # NOTE: this is KITTI-specific
        self.points = homogenization_by_minimal_pruning(self.points)

    def _load_timestamps(self, lidar_ts_path: str, images_ts_path: str):
        """Loads LiDAR timestamps (.txt), and camera timestamps (.txt) data in KITTI format

        Args:
            lidar_ts_path: containing LiDAR point clouds in KITTI (.bin) format.
            images_ts_path: equivalent to ^^, but only requires subdirectory `data`.

        Returns:
            Timestamps of
                - LiDAR samples
                - camera samples
                in ns, both relative the first sample in time, which can be either a LiDAR or camera sample.
        """
        #
        with open(lidar_ts_path, "r") as f:
            lidar_ts = f.readlines()[
                self.start_frame_id : self.start_frame_id + self.num_frames
            ]
        with open(images_ts_path, "r") as f:
            image_ts = f.readlines()[
                self.start_frame_id : self.start_frame_id + self.num_frames
            ]

        #  convert timestamps (e.g '2011-10-03 12:57:44.520977408') to nanosec relative to first timestamp
        ref_ts = np.min(
            np.array(
                [np.datetime64(image_ts[0], "ns"), np.datetime64(lidar_ts[0], "ns")],
                dtype="datetime64[ns]",
            )
        )
        image_ts_ns = np.array(image_ts[0:], dtype="datetime64[ns]") - np.datetime64(
            ref_ts
        )

        #  gives np.array of np.timedelta64[ns]
        lidar_ts_ns = np.array(lidar_ts[0:], dtype="datetime64[ns]") - np.datetime64(
            ref_ts
        )
        return (
            lidar_ts_ns.astype("int64"),
            image_ts_ns.astype("int64"),
        )

    def _interpolate_pose(self, T1: np.ndarray, T2: np.ndarray, t1, t2, t_out):
        """Interpolates pose T1 linearly to T2 based on given timestamps.

        Reference:
            - Follows principles from MDPCalib's pose_synchronizer.cpp
            - https://github.com/robot-learning-freiburg/MDPCalib
        """
        # linear interpolation
        dt = (t_out - t1) / (t2 - t1)
        t = np.zeros((3, 1), dtype=np.float32)
        t[:, 0] = T1[0:3, 3] + dt * (T2[0:3, 3] - T1[0:3, 3])

        # spherical linear interpolation
        R1 = quaternion.from_rotation_matrix(T1[0:3, 0:3])
        R2 = quaternion.from_rotation_matrix(T2[0:3, 0:3])
        R1_to_ref = R1.conjugate() * quaternion.slerp(R1, R2, t1, t2, t_out)
        R = R1_to_ref * R1
        R_mat = quaternion.as_rotation_matrix(R)

        # construct interpolated transformation
        interpolated_T = np.copy(
            T1
        )  # get bottom row homogeneous and same shape (i.e 4x4)
        interpolated_T[0:3, 0:3] = R_mat  # set rotation 3x3
        interpolated_T[0:3, 3] = t[:, 0]  # set translation 3x1
        return interpolated_T

    def _interpolate_points(
        self,
        lidar_ts_path: str,
        images_ts_path: str,
        poses_path: str,
        num_frames: int = None,
    ):
        """Linear interpolation of LiDAR poses/scans to match camera's reference timestamps.

        Note: this simple implementation assumes you never have two camera samples in between the
            same LiDAR pose pair. Therefore, this method always drops one frame and offset tells us which.

        Reference:
            - Follows principles from MDPCalib's pose_synchronizer.cpp
            - https://github.com/robot-learning-freiburg/MDPCalib

        Returns:
            offset telling us that first camera could be interpolated (i.e offset==1), otherwise
                last camera frame required extrapolation.
        """
        # NOTE: interpolation will overwrite points
        # FIXME: currently consecutive frames are assumed, and poses[0] is assumed to correspond to START_FRAME_IDX
        lidar_ts_ns, cam_ts_ns = self._load_timestamps(
            lidar_ts_path=lidar_ts_path, images_ts_path=images_ts_path
        )
        poses = np.load(poses_path)

        interpolated_frames = list()
        prev_li_ts = lidar_ts_ns[0]
        offset = 0
        if cam_ts_ns[offset] < prev_li_ts:
            offset = 1  # bec. impossible to interpolate without prev
        i = 1
        for ref_ts in cam_ts_ns[offset:]:
            print("ref_ts:", ref_ts)
            if not (prev_li_ts <= ref_ts <= lidar_ts_ns[i]):
                raise Exception(
                    "No interpolation possible for a given camera frame because multiple camera samples were found for same pose pair."
                )

            T1 = poses[i - 1]
            T2 = poses[i]
            interpolated_T = self._interpolate_pose(
                T1=T1, T2=T2, t1=prev_li_ts, t2=lidar_ts_ns[i], t_out=ref_ts
            )

            # transform LiDAR scan from T1 to T_interpolated
            frame_points = self.points[i - 1][:, 0:3]  # (n x 3)
            T_rel = np.linalg.inv(interpolated_T) @ T1

            hom_points = np.concatenate(
                (
                    frame_points.T,
                    np.ones(shape=(1, frame_points.T.shape[1]), dtype=np.float32),
                ),
                axis=0,
            )  # (4 x n)

            interpolated_points = T_rel @ hom_points
            interpolated_frames.append(interpolated_points[0:3, :].T)

            prev_li_ts = lidar_ts_ns[i]
            i += 1
            if not i < len(lidar_ts_ns):
                break  # early break means that offset = 0, since there is no next pose for last cam

        # remove first or last frame using offset
        print("offset:", offset)
        self.frame_ids = self.frame_ids[offset : len(self.frame_ids) - (1 - offset)]

        # else, all remaining frames (so N-1) are used for rest of calibration
        self.points = interpolated_frames

        print(
            f"[INFO] LiDAR interpolated (offset = {offset}). One frame has been dropped. Don't forget to reload image frames using the updated frame ids."
        )
