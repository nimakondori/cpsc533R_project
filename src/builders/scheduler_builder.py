from torch import optim
from torch.optim.optimizer import Optimizer
# from src.core.schedulers import CustomScheduler
from copy import deepcopy


# TODO: See if you need to apply a custom scheduler
SCHEDULERS = {
    'multi': optim.lr_scheduler.MultiStepLR,
    'reduce_lr_on_plateau': optim.lr_scheduler.ReduceLROnPlateau,
    # 'custom': CustomScheduler
}

def build(config, optimizer, logger):
    if 'lr_schedule' not in config:
        logger.warn('No scheduler is specified.')
        return None

    schedule_config = deepcopy(config['lr_schedule'])
    # "multi" here is the default return value if lr_scheuder is not specified in the config
    scheduler_name = schedule_config.pop('name', 'multi')
    schedule_config['optimizer'] = optimizer

    # TODO: Replace with try except block maybe?
    if scheduler_name in SCHEDULERS:
        scheduler = SCHEDULERS[scheduler_name](**schedule_config)
    else:
        logger.error(
            'Specify a valid scheduler name among {}.'.format(SCHEDULERS.keys())
        ); exit()

    logger.infov('{} scheduler is built.'.format(scheduler_name.upper()))
    return scheduler
