import argparse


def depth_args():
    """Function to parse script arguments which are common in the depth predictions scripts."""
    parser = argparse.ArgumentParser(description="Fine registration.")
    parser.add_argument(
        "--start_frame_id",
        type=int,
        required=True,
        help="ID of the frame to start from, which is assumed to correspond to "
        "first LiDAR frame in data directory used by KISS-SLAM.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=7,
        help="Number of frames to compute to average the NID.",
    )
    args = parser.parse_args()
    return args
