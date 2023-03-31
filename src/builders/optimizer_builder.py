import torch
from copy import deepcopy


OPTIMIZERS = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam
}


def build(config, model, logger):
    config = deepcopy(config)
    optimizer_name = config.pop('name')    
    config['params'] = model.parameters()
    optimizer = OPTIMIZERS[optimizer_name](**config)

    logger.info('{} opimizer is built.'.format(optimizer_name.upper()))
    return optimizer