# from torch_geometric.loader import DataListLoader, DataLoader
import os
import platform
from torch.utils.data import DataLoader
from copy import deepcopy


def build(datasets, train_config, logger, use_data_parallel=False):
    # Get data parameters
    config = deepcopy(train_config)
    batch_size = config.pop('batch_size')
    num_workers = config.pop('num_workers')


    # check the operating system and update the number of workers
    if platform.system() == "Windows":
        num_workers = 0
        logger.info(f"Identified Windows OS, Setting num_workers to {num_workers} for optimal performance.")
    elif platform.system() == "Linux":
        num_workers = min(8, os.cpu_count())
        logger.info(f"Identified Linux OS, Setting num_workers to {num_workers} for optimal performance.")
    else:
        logger.info("Unknown operating system, keeping the config num_workers.")


    # Load datalodaers for each mode
    dataloaders = {}
    for dataset_name in datasets.keys():
        dataloaders[dataset_name] = {}
        for mode in ['train', 'val', 'test']:
            shuffle = True if mode == 'train' else False
            drop_last = True if mode in ['train', 'val'] else False

            # if use_data_parallel:
            #     dataloader = DataListLoader(datasets[dataset_name][mode],
            #                                 batch_size=batch_size,
            #                                 shuffle=shuffle,
            #                                 num_workers=num_workers,
            #                                 drop_last=drop_last)
            # else:
            dataloader = DataLoader(datasets[dataset_name][mode],
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    num_workers=num_workers,
                                    drop_last=drop_last)
            dataloaders[dataset_name][mode] = dataloader

    logger.info("Dataloders are created.")

    return dataloaders

