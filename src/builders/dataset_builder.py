from torchvision import transforms
from src.core.datasets import LVIDLandmark
from src.utils.util import normalization_params
from copy import deepcopy


DATASETS = {
    'lvidlandmark': LVIDLandmark,    
}


def build(data_config, logger):
    # Get data parameters
    configs = deepcopy(data_config)

    # Create the datasets for each mode
    datasets = {}   
    for config in configs:
        config = config["dataset"]
        dataset_name = config.pop('name')
        datasets[dataset_name] = {}
        transform_config = config.pop('transform')

        if dataset_name not in DATASETS:
            logger.error('No data named {}'.format(dataset_name))

        
        for mode in ['train', 'val', 'test']:
            transform = compose_transforms(transform_config)
            datasets[dataset_name][mode] = DATASETS[dataset_name](**config,
                                                                  mode=mode,
                                                                  logger=logger,
                                                                  transform=transform,
                                                                  frame_size=transform_config['image_size'])

    return datasets


def compose_transforms(transform_config):
    # check if we need mode ?
    mean, std = normalization_params()
    image_size = transform_config['image_size']
    make_gray = transform_config.get('make_gray', False)

    # Do we need gamma correction in these or not?
    transforms_list = [transforms.Resize((image_size, image_size))]

    if make_gray:
        transforms_list.append(transforms.Grayscale())

    transform = transforms.Compose(transforms_list)
    return transform

