# The LVID dataloader is inspired by the dataloader from Masoud Mokhtari et. al. HiGNN paper

import os
import random
import ast
import pickle
import bz2
import scipy.io as sio
import pandas as pd
import numpy as np
import torch
from abc import ABC
from torch.utils.data import Dataset
from torchvision.transforms.functional import hflip


NUM_PREFETCH = 10
RANDOM_SEED = 7

class LVIDLandmark(Dataset, ABC):
    def __init__(self,
                 data_dir,
                 metadata_dir,
                 mode,
                 logger=None,
                 transform=None,
                 frame_size=128,
                #  `average_coords`=None,
                 flip_p=0.0):

        super().__init__()

        # if average_coords is None:
        #     # These numbers are obtained using the average_landmark_locations.py script
        #     self.average_coords = [[99.99, 112.57], [142.71, 90.67], [151.18, 86.25], [91.81, 117.91]]
        # else:
        #     self.average_coords = average_coords

        # Read the data CSV file
        self.data_info = pd.read_csv(metadata_dir)

        # Rename the index column so the processed data can be tracked down later
        self.data_info = self.data_info.rename(columns={'Unnamed: 0': 'db_indx'})

        # Get the correct data split. Data split was applied during the preprocessing
        self.data_info = self.data_info[self.data_info.split == mode]

        if logger is not None:
            logger.info(f'#{mode}: {len(self.data_info)}')

        # Add root directory to file names to create full path to data
        self.data_info['cleaned_path'] = self.data_info.apply(lambda row: os.path.join(data_dir, row['file_name']),
                                                              axis=1)

        # Other required attributes
        self.mode = mode
        self.logger = logger
        self.transform = transform
        self.flip_p = flip_p
        self.frame_size = frame_size

    def __getitem__(self, idx):
        data_item = {}        
        data = self.data_info.iloc[idx]

        # Unpickle the data
        pickle_file = bz2.BZ2File(data['cleaned_path'], 'rb')
        mat_contents = pickle.load(pickle_file)
        cine = mat_contents['resized'] # 224x224xN

        # Extracting the ED frame
        if data['d_frame_number'] > cine.shape[-1]:
            ed_frame = cine[:, :, -1]
        else:
            ed_frame = cine[:, :, data['d_frame_number']-1]

        # ed_frame shape = (224,224),  transform to torch tensor with shape (1,1,resized_size,resized_size)
        orig_size = ed_frame.shape[0]
        ed_frame = torch.tensor(ed_frame, dtype=torch.float32).unsqueeze(0) / 255  # (1, 224,224)
        ed_frame = self.transform(ed_frame).unsqueeze(0)  # (1, 1, frame_size, frame_size)

        # Extract landmark coordinates
        coords = self.extract_coords(data, orig_size)

        if random.uniform(0, 1) <= self.flip_p and self.mode == "train":
            coords[:, 1] = self.frame_size - coords[:, 1] - 1
            ed_frame = hflip(ed_frame)
        
        # Add the echo frame to data_item
        data_item["x"] = ed_frame.squeeze(0)

        # Create labels from the coordinates
        data_item["y"] = torch.from_numpy(coords)
        data_item["valid_labels"] = torch.ones_like(data_item["y"])

        # Get the scale for each pixel in mm/pixel
        deltaY = data['DeltaY'] * orig_size / self.frame_size
        deltaX = data['DeltaX'] * orig_size / self.frame_size
        data_item["pix2mm_x"] = torch.tensor(deltaX * 10, dtype=torch.float32)  # in mm
        data_item["pix2mm_y"] = torch.tensor(deltaY * 10, dtype=torch.float32)  # in mm

        if self.mode != 'train':  
            keys = ['db_indx', "PatientID", "StudyDate", "SIUID", "LV_Mass", "BSA", "file_name"]
            data_item.update(data[keys].to_dict())

        return data_item

    def __len__(self):
        return len(self.data_info)

    def extract_coords(self, df, orig_frame_size):

        # get all landmark coordinates, select the four we need
        LVIDd_coordinate = np.round(np.array(ast.literal_eval(df['LVID'])) * self.frame_size / orig_frame_size).astype(int)
        IVS_coordinates = np.round(np.array(ast.literal_eval(df['IVS'])) * self.frame_size / orig_frame_size).astype(int)
        LVPW_coordinates = np.round(np.array(ast.literal_eval(df['LVPW'])) * self.frame_size / orig_frame_size).astype(int)

        # Note that the coordinates are saved in (h, w) convention. in order: LVID_top, LVID_bot, LVPW, IVS
        coords = []
        coords.append([LVIDd_coordinate[1] - 1, LVIDd_coordinate[0] - 1])
        coords.append([LVIDd_coordinate[3] - 1, LVIDd_coordinate[2] - 1])
        coords.append([LVPW_coordinates[3] - 1, LVPW_coordinates[2] - 1])
        coords.append([IVS_coordinates[1] - 1, IVS_coordinates[0] - 1])
        coords = np.array(coords)

        return coords
