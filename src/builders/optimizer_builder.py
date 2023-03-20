import torch
from copy import deepcopy


OPTIMIZERS = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam
}


def build(config, models, logger):
    config = deepcopy(config)
    optimizer_name = config.pop('name')

    # TODO: This should be replaced with proper models from config 
    config['params'] = models['embedder'].parameters()
    optimizer = OPTIMIZERS[optimizer_name](**config)

    logger.info('{} opimizer is built.'.format(optimizer_name.upper()))
    return optimizer