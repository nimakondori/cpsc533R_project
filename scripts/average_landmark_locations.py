# source: Mokhtari et. al. HiGNN
import pandas as pd
import argparse
import ast
import numpy as np
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv_path', required=True,
                        help="Path to CSV file containing EF labels for EchoNet")
    args = parser.parse_args()

    data_info = pd.read_csv(args.data_csv_path)
    data_info = data_info[data_info.split == 'train']

    for idx in range(data_info.shape[0]):
        data = data_info.iloc[idx]

        LVIDd_coordinate = np.round(np.array(ast.literal_eval(data['LVID']))).astype(float)
        IVS_coordinates = np.round(np.array(ast.literal_eval(data['IVS']))).astype(float)
        LVPW_coordinates = np.round(np.array(ast.literal_eval(data['LVPW']))).astype(float)

        new_coords = []
        new_coords.append([LVIDd_coordinate[1] - 1, LVIDd_coordinate[0] - 1])
        new_coords.append([LVIDd_coordinate[3] - 1, LVIDd_coordinate[2] - 1])
        new_coords.append([LVPW_coordinates[3] - 1, LVPW_coordinates[2] - 1])
        new_coords.append([IVS_coordinates[1] - 1, IVS_coordinates[0] - 1])
        new_coords = np.array(new_coords)

        coords = coords + new_coords if idx !=0 else new_coords

    coords = coords / data_info.shape[0]
    print('Average coordinates are: {}'.format(coords))