import os
import yaml
import argparse
import pandas as pd

import numpy as np

from modules.dataset import Dataset
from modules.pipeline import Pipeline

from modules.preprocessing import fibonacci_sphere, get_error_transform

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Targetless calibration pipeline")
    parser.add_argument(
        "--start_frame_id",
        type=int,
        required=True,
        help="ID of the frame to start from, which is assumed to correspond to first LiDAR frame in data directory used by KISS-SLAM.",
    )
    parser.add_argument(  # NOTE: subsequent IDs are consecutive integers, i.e., i+1, i+2, i+3
        "--num_frames",
        type=int,
        default=8,
        help="Number of frames to compute to average the NID.",
    )
    parser.add_argument(
        "--interpolate",
        action="store_true",
        default=False,
        help="Will linearly interpolate LiDAR point clouds based on KISS-SLAM pose estimates.",
    )
    parser.add_argument(
        "--rotation_only",
        action="store_true",
        default=False,
        help="Will only parameterize the rotation component and keep translation fixed at initial guess.",
    )
    parser.add_argument(
        "--add_noise",
        action="store_true",
        default=False,
        help="Adds the given amount of noise.",
    )
    parser.add_argument(
        "--is_example",
        action="store_true",
        default=False,
        help="Will prints the difference relative to the initial guess, such that it's the calibration error when this corresponds to the ground truth and the `add_noise` argument is True.",
    )
    args = parser.parse_args()
    config = yaml.safe_load(open("config/config.yaml"))

    ######################
    # CONFIGURE PIPELINE #
    ######################

    # Load and preprocess LiDAR data
    dataset = Dataset(num_frames=args.num_frames, start_frame_id=args.start_frame_id)
    dataset.load_lidar_data(
        os.path.join(
            config["data_dir"],
            "lidar",
        ),
    )
    dataset.preprocess_points(
        interpolate=args.interpolate,
        lidar_ts_path=os.path.join(config["data_dir"], "lidar", "timestamps.txt"),
        images_ts_path=os.path.join(config["data_dir"], "images", "timestamps.txt"),
        poses_path=config["data_poses"],
        num_frames_after_interpolation=1,
    )

    # NOTE: this allows to batch the LiDAR projection etc.
    model = "UNIDEPTH"
    conf_th = 0
    dataset.load_camera_data(
        base_path=os.path.join(config["data_dir"], model.lower()),
        model=model,
        conf_th=conf_th,
    )
    # NOTE: will be replaced with pykitti function (see eval_calib.py) (TODO)
    dataset.load_calibration(
        images_dir=os.path.join(config["data_dir"], "images", "data"),
        camera_calibration_path=os.path.join(
            config["data_dir"], "calib_cam_to_cam.txt"
        ),
        extrinsic_calibration_path=os.path.join(
            config["data_dir"], "calib_velo_to_cam.txt"
        ),
        camera_id=2,
    )
    pipeline = Pipeline(dataset=dataset, optimizer_tolerance=1e-5)

    # NOTE: replace T_init with your initial guess
    T_init = dataset.T_gt  # 4x4 extrinsic LiDAR to camera transform
    #        ^^^^^^^^^^^^, TODO: substitute this with np.load of your initial guess
    if args.add_noise:
        # pick random direction to apply error from fib. sphere samples
        num_points = 10
        rand_i = np.random.randint(low=0, high=num_points)
        fib_axes = fibonacci_sphere(num_points=num_points, show_plot=False)[rand_i]
        T_error = get_error_transform(R_error=5, t_error=0.5, direction=fib_axes)
        #                                     ^          ^^, TODO: substitue desired errors here
        T_init = T_init @ T_error

    ################
    # OPTIMIZATION #
    ################
    
    # optimization
    pipeline.optimize(T_init=T_init, rotation_only=args.rotation_only)
    if args.is_example:
        # prints calibration errors
        # TODO: you can get rid of the enclosing if statement for usage besides example
        _ = pipeline.evaluate()

    # save the optimized extrinsic camera-LiDAR transformation
    np.save("T_star.npy", pipeline.T_star)
