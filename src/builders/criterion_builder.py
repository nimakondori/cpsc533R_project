import torch.nn as nn
from src.core.criterion import WeightedBCE, WeightedBCEWithLogitsLoss, MSE, ExpectedLandmarkMSE, HeatmapMSELoss, MAE
from copy import deepcopy


CRITERIA = {
    'mse': MSE,  # loss for coordinates
    'mae': MAE,  # loss for coordinates
    'bce': WeightedBCE,  # per-pixel loss for heatmaps
    'HeatmapMse': HeatmapMSELoss,  # per-pixel loss for heatmaps
    'WeightedBceWithLogits': WeightedBCEWithLogitsLoss,  # per-pixel loss for heatmaps
    'ExpectedLandmarkMse': ExpectedLandmarkMSE, # loss for coordinates using heatmaps
}


def build(config, logger):
    # TODO: Refactor this code
    config = deepcopy(config)

    criteria = dict()
    for criterion_name, criterion_config in config.items():
        criteria[criterion_name] = CRITERIA[criterion_name](**criterion_config)

        logger.infov('{} criterion is built.'.format(criterion_name.upper()))
    return criteria