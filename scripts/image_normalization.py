from utils.support import pkl_load, jsn_dump

from pathlib import Path
import os
import numpy as np
import cv2
from tqdm import tqdm
import sys


def main(cfg):
    results_dir = Path(cfg["output"]) / "image_normalization"
    os.makedirs(results_dir, exist_ok=True)
    jsn_dump(cfg, results_dir / "config.json")

    identity_df = pkl_load(Path(cfg["output"]) / "crossval_split" / "identity_df.pkl")
    img_paths = identity_df["cew_img"].values

    # mean
    sum_ = 0
    count = 0
    for img_path in tqdm(img_paths, file=sys.stdout, total=len(img_paths)):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = img / 255
            sum_ += img.sum()
            count += img.shape[0] * img.shape[1]
    mean = np.round(sum_ / count, 3)

    # std
    diff_squared = 0
    for img_path in tqdm(img_paths, file=sys.stdout, total=len(img_paths)):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = img / 255
            diff_squared += ((img - mean) ** 2).sum()
    std = np.round(np.sqrt(diff_squared / count), 3)

    with open(results_dir / "normalization_params.txt", "w") as file:
        file.write(f"mean = {mean}")
        file.write("\n")
        file.write(f"std = {std}")


if __name__ == "__main__":
    config = {
        "output": "../output"
    }
    main(config)
