import os
import glob
from pathlib import Path
import numpy as np 
import pandas as pd


# DF_PATH = Path("data\LVOT_Cleaned_all.csv")
# DATA_PATH = Path("data\LVOT_cleaned\*.mat")
# COLUMN_NAME = "file_name" 
DF_PATH = Path("data\lv_plax2_cleaned_info_landmark_gt_filtered_es_cleaned.csv")
DATA_PATH = Path("data\LVID\LV_PLAX2_cleaned\*.pbz2")
COLUMN_NAME = "file_name" 


if __name__ == "__main__":

    extract_indices, extract_paths = None, None

    original_df = pd.read_csv(DF_PATH)
    filenames = [os.path.basename(x) for x  in glob.glob(str(DATA_PATH))]

    short_df = original_df[original_df[COLUMN_NAME].isin(filenames)]

    save_dir = DF_PATH.parent
    short_df.to_csv(os.path.join(str(save_dir), DF_PATH.name.split(".")[0] + "_short." + DF_PATH.name.split(".")[1]))
