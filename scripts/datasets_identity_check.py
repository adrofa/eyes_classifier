import os
import sys
from pathlib import Path
import pandas as pd
import cv2
import logging


def get_logger(logger_name):
    """Initializes logger and prints 1st log-message into stdout."""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(logger_name)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("Module started.")

    return logger


def gen_dataset_df(data_path_):
    """Collect table with full paths to image files, their sizes in bytes
    and images themselves as numpy array.

    Args:
        data_path_ (str): path to folder with images.

    Returns:
        dataset_df_ (pandas.DataFrame): DataFrame with 3 columns: img_path, size, img.
    """
    dataset_df_ = pd.DataFrame()

    for dir_name, _, file_names in os.walk(data_path_):
        for filename in file_names:
            img_path = Path(os.path.join(dir_name, filename))
            size_ = img_path.stat().st_size
            dataset_df_ = dataset_df_.append({
                "img_path": img_path,
                "size": size_,
                "img": cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE),
            }, ignore_index=True)

    dataset_df_["size"] = dataset_df_["size"].astype(int)

    return dataset_df_


if __name__ == "__main__":
    logger = get_logger("datasets_identity_check")

    external_path = Path("../data/dataset_B_Eye_Images/")
    internal_path = Path("../data/EyesDataset")

    external_df = gen_dataset_df(external_path)
    logger.info("External dataset generated.")
    internal_df = gen_dataset_df(internal_path)
    logger.info("Internal dataset generated.")

    identity_df = pd.DataFrame()
    for size in external_df["size"].unique():
        ext_df = external_df[external_df["size"] == size]
        int_df = internal_df[internal_df["size"] == size]

        for _, ext_row in ext_df.iterrows():
            pair = None
            for _, int_row in int_df.iterrows():
                if (ext_row["img"] == int_row["img"]).all():
                    pair = int_row["img_path"]
                    break

            identity_df = identity_df.append({
                "external_img": ext_row["img_path"],
                "internal_img": pair
            }, ignore_index=True)
    logger.info("Identity check completed")

    if (~identity_df["internal_img"].isnull()).sum() == len(internal_df):
        logger.info("Datasets are identical.")
    else:
        logger.info("Datasets are NOT identical.")
