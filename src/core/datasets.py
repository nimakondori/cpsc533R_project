# The LVOT dataloader is inspired by the dataloader form Mohammad Jafari's dataloader from U-Land paper
# The LVID dataloader is inspired by the dataloader from Masoud Mokhtari et. al. HiGNN paper

import os
import random
import ast
import cv2
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


class LVOTLandmark(Dataset, ABC):
    def __init__(self, dataset_path, img_res, augment, deploy):
        self.dataset_path = dataset_path
        self.img_res = tuple(img_res)
        self.augment = augment

        df_train = pd.read_csv(os.path.join(self.dataset_path,
                                            'LVOT_train_extra_info_refined_based_on_landmark_presence.csv'))
        df_valid = pd.read_csv(os.path.join(self.dataset_path,
                                            'LVOT_val_extra_info_refined_based_on_landmark_presence.csv'))
        df_test = pd.read_csv(os.path.join(self.dataset_path,
                                           'LVOT_test_extra_info_refined_based_on_landmark_presence.csv'))

        train_paths = df_train['Path'].tolist()
        valid_paths = df_valid['Path'].tolist()
        test_paths = df_test['Path'].tolist()

        train_paths = [os.path.join(self.dataset_path, 'LVOT_cleaned', path.split('\\')[-1]) for path in train_paths]
        valid_paths = [os.path.join(self.dataset_path, 'LVOT_cleaned', path.split('\\')[-1]) for path in valid_paths]
        test_paths = [os.path.join(self.dataset_path, 'LVOT_cleaned', path.split('\\')[-1]) for path in test_paths]

        train_pixel_x = df_train['PhysicalDeltaX'].tolist()
        valid_pixel_x = df_valid['PhysicalDeltaX'].tolist()
        test_pixel_x = df_test['PhysicalDeltaX'].tolist()

        train_pixel_y = df_train['PhysicalDeltaY'].tolist()
        valid_pixel_y  = df_valid['PhysicalDeltaY'].tolist()
        test_pixel_y  = df_test['PhysicalDeltaY'].tolist()

        self.train_patients = list(zip(train_paths, train_pixel_x, train_pixel_y))
        self.valid_patients = list(zip(valid_paths, valid_pixel_x, valid_pixel_y))
        self.test_patients = list(zip(test_paths, test_pixel_x, test_pixel_y))

        print('#train:', len(self.train_patients))
        print('#valid:', len(self.valid_patients))
        print('#test:', len(self.test_patients))
        print('Consistency check - First valid sample:', self.valid_patients[0])
        print('Consistency check - First test sample:', self.test_patients[0])

        # data augmentation configuration
        data_gen_args = dict(rotation_range=augment['AUG_ROTATION_RANGE_DEGREES'],
                             width_shift_range=augment['AUG_WIDTH_SHIFT_RANGE_RATIO'],
                             height_shift_range=augment['AUG_HEIGHT_SHIFT_RANGE_RATIO'],
                             shear_range=augment['AUG_SHEAR_RANGE_ANGLE'],
                             zoom_range=augment['AUG_ZOOM_RANGE_RATIO'],
                             fill_mode='constant',
                             cval=0.,
                             data_format='channels_last')
        self.datagen = ImageDataGenerator(**data_gen_args)

    def _get_paths(self, stage):
        if stage == 'train':
            return self.train_patients
        elif stage == 'valid':
            return self.valid_patients
        elif stage == 'test':
            return self.test_patients

    @staticmethod
    def gamma_corr(imgs, Gamma_Range):
        flag = 0
        if np.max(imgs) > 1:
            imgs = imgs / 255.0
            flag = 1
        gamma = random.uniform(Gamma_Range[0], Gamma_Range[1])
        imgs = np.power(imgs, gamma)
        if flag:
            imgs = imgs * 255
            imgs = imgs.astype(np.uint8)
        return imgs

    @staticmethod
    def randomCrop(imgs, max_width, max_height):
        width = random.randint(0, np.int(max_width))
        height = random.randint(0, np.int(max_height))
        x = random.randint(0, imgs.shape[1] - width)
        y = random.randint(0, imgs.shape[0] - height)
        imgs[y:y + height, x:x + width, :] = 0
        return imgs

    # multithreading data loading
    # @background(max_prefetch=NUM_PREFETCH)
    def get_random_batch(self, batch_size=1, stage='train'):
        paths = self._get_paths(stage)
        num = len(paths)
        num_batches = num // batch_size
        for i in range(num_batches):
            batch_paths = random.sample(paths, batch_size)
            imgs, gt, patient_paths = self._get_batch(batch_paths, stage)
            yield imgs, gt, patient_paths

    def get_iterative_batch(self, batch_size=1, stage='test'):
        paths = self._get_paths(stage)
        num = len(paths)
        num_batches = num // batch_size
        start_idx = 0
        for i in range(num_batches):
            batch_paths = paths[start_idx:start_idx + batch_size]
            imgs, gt, patient_paths = self._get_batch(batch_paths, stage)
            start_idx += batch_size
            yield imgs, gt, patient_paths

    def _get_batch(self, paths_batch, stage):
        imgs = []
        gts = []
        paths = []
        for path in paths_batch:

            # load matfiles
            mat_contents = sio.loadmat(path[0])
            cine = mat_contents['cine']
            img_LVOT = cine[:,:,mat_contents['lvot_frame'][0][0]-1]
            img_LVOT = cv2.resize(img_LVOT, (self.img_res[0], self.img_res[1]))

            LVOT_coordinate = mat_contents['lvot_label'][0]
            gt_LVOT = np.zeros([cine.shape[0],cine.shape[1]])
            gt_LVOT =  cv2.circle(gt_LVOT, (LVOT_coordinate[0]-1, LVOT_coordinate[1]-1), self.heatmap_radius, 1, -1)
            gt_LVOT =  cv2.circle(gt_LVOT, (LVOT_coordinate[2]-1, LVOT_coordinate[3]-1), self.heatmap_radius, 1, -1)
            gt_LVOT =  cv2.resize(gt_LVOT, (self.img_res[0], self.img_res[1]))

            img_LVOT = img_LVOT[:, :, np.newaxis]
            gt_LVOT = gt_LVOT[:, :, np.newaxis]

            if stage == 'train':
                # if in training stage => augment the dataset by transform, crop, gamma correction
                transform = self.datagen.get_random_transform(img_shape=self.img_res)
                img_LVOT = self.datagen.apply_transform(img_LVOT, transform)
                img_LVOT = self.gamma_corr(img_LVOT, self.augment['AUG_GAMMA'])
                img_LVOT = self.randomCrop(img_LVOT, img_LVOT.shape[1] / 4.0, img_LVOT.shape[0] / 4.0)
                gt_LVOT = self.datagen.apply_transform(gt_LVOT, transform)

            # round ground truth to be 0 or 1
            gt_LVOT = np.round(gt_LVOT)

            # create a list of data in the batch
            imgs.append(img_LVOT)
            gts.append(gt_LVOT)
            paths.append(path)

        # convert data to numpy array and float64 for training, normalize images and labels to be between 0 and 1
        imgs = np.array(imgs)
        gts = np.array(gts)
        imgs = imgs.astype('float64')
        gts = gts.astype('float64')
        imgs = imgs / 255.0
        return imgs, gts , paths

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
        # Get the data at index
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
        ed_frame = self.transform(ed_frame).unsqueeze(0)  # (1,1,frame_size,frame_size)

        # Extract landmark coordinates
        coords = self.extract_coords(data, orig_size)

        if random.uniform(0, 1) <= self.flip_p and self.mode == "train":
            coords[:, 1] = self.frame_size - coords[:, 1] - 1
            ed_frame = hflip(ed_frame)
        
        # Add the echo frame to data_item
        data_item["x"] = ed_frame

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
    


class LVOTLandmark2(Dataset, ABC):
    def __init__(self, 
                 data_dir,
                 metadata_dir,
                 mode,
                 transform=None,
                 frame_size=128,
                 heatmap_radius=7,
                 logger=None):
        
        self.data_dir = data_dir
        self.data_info = pd.read_csv(metadata_dir)
        # Rename the index column so the processed data can be tracked down later
        self.data_info = self.data_info.rename(columns={'Unnamed: 0': 'db_indx'})

        # Get the correct data split. Data split was applied during the preprocessing
        self.data_info = self.data_info[self.data_info.split == mode]

         # Add root directory to file names to create full path to data
        self.data_info['cleaned_path'] = self.data_info.apply(lambda row: os.path.join(data_dir, row['file_name']),
                                                              axis=1)



        # self.patients_data = list(zip(data_paths, data_pixel_x, data_pixel_y))
        self.mode = mode
        self.logger = logger
        self.transform = transform
        self.frame_size = frame_size
        self.heatmap_radius = heatmap_radius

        if logger is not None:
            logger.info(f'#{mode}: {len(self.data_info)}')

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        data_item = {}
        # Get the data at index
        data = self.data_info.iloc[idx]
        # path, data_pixel_x, data_pixel_y = self.data_info.iloc[idx]
        path = data["cleaned_path"]
        
        mat_contents = sio.loadmat(data["cleaned_path"])
        cine = mat_contents['cine']
        img_LVOT = cine[:,:,mat_contents['lvot_frame'][0][0]-1]
        img_LVOT = cv2.resize(img_LVOT, (self.frame_size, self.frame_size))

        LVOT_coordinate = mat_contents['lvot_label'][0]
        gt_LVOT = np.zeros([cine.shape[0],cine.shape[1]])
        gt_LVOT =  cv2.circle(gt_LVOT, (LVOT_coordinate[0]-1, LVOT_coordinate[1]-1), self.heatmap_radius, 1, -1)
        gt_LVOT =  cv2.circle(gt_LVOT, (LVOT_coordinate[2]-1, LVOT_coordinate[3]-1), self.heatmap_radius, 1, -1)
        gt_LVOT =  cv2.resize(gt_LVOT, (self.frame_size, self.frame_size))

        # TODO: fix the from numpy and the dimensions of the data
        img_LVOT = img_LVOT[:, :, np.newaxis]
        data_item["x"] = torch.from_numpy(img_LVOT)
        gt_LVOT = gt_LVOT[:, :, np.newaxis]
        data_item["gt_LVOT"] = torch.from_numpy(gt_LVOT)
        data_item["path"] = path
        data_item["y"] =  torch.from_numpy(LVOT_coordinate.astype(int))

        return data_item



