import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
import torch
from PIL import Image

from modules.script_utils import depth_args
from unidepth.models import UniDepthV2

from unidepth.utils.camera import Pinhole


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary.

    (not my code, taken from pykitti)
    Source: https://github.com/utiasSTARS/pykitti/
    """
    data = {}

    with open(filepath, "r") as f:
        for line in f.readlines():
            try:
                key, value = line.split(":", 1)
            except ValueError:
                key, value = line.split(" ", 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


if __name__ == "__main__":
    args = depth_args()
    config = yaml.safe_load(open("config/config.yaml"))

    # prepare image paths
    frame_ids = [
        f"{id:010d}"
        for id in range(args.start_frame_id, args.start_frame_id + args.num_frames)
    ]
    out_dir = os.path.join(config["data_dir"], "unidepth")
    os.makedirs(out_dir, exist_ok=True)
    paths = [
        (
            os.path.join(config["data_dir"], "images", "data", frame_id + ".png"),
            os.path.join(out_dir, frame_id + "_depth.npy"),
            os.path.join(out_dir, frame_id + "_uncertainty.npy"),
        )
        for frame_id in frame_ids
    ]

    # load the UniDepth V2 model of specified size
    type_ = "l"  # NOTE: available types: s, b, l
    name = f"unidepth-v2-vit{type_}14"
    model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")
    model.resolution_level = 9  # NOTE: in range [0, 10) (so 9 is max)
    model.interpolation_mode = "bilinear"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # load camera matrix (KITTI) of the left RGB camera (after rectification)
    calib_data = read_calib_file("data/calib_cam_to_cam.txt")
    P_rect = calib_data["P_rect_02"].reshape((3, 4))
    K_rect = P_rect[:3, :3]
    intrinsics_torch = torch.from_numpy(K_rect)
    camera = Pinhole(K=intrinsics_torch.unsqueeze(0))

    for (
        image_path,
        depth_path,
        uncertainty_path,
    ) in paths:
        # Source: https://github.com/lpiccinelli-eth/UniDepth/blob/main/scripts/demo.py
        rgb = np.array(Image.open(image_path))
        rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)

        # Predict
        predictions = model.infer(rgb_torch, camera)

        # Get pred
        depth_pred = predictions["depth"].squeeze().cpu().numpy()
        uncertainty_pred = predictions["confidence"].squeeze().cpu().numpy()

        # saves all thre to disk
        np.save(depth_path, depth_pred)  # shape is same as input
        np.save(uncertainty_path, uncertainty_pred)  # shape is same as input

        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
