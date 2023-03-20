from src.core.models import UMMT, CNN
from copy import deepcopy


MODELS = {
    'umtt': UMMT,
    'cnn': CNN
}


def build(config, logger):

    config = deepcopy(config)
    _ = config.pop('checkpoint_path')

    # Build a model
    model = dict()
    for model_key in config:
        model_name = config[model_key].pop('name')
        model[model_key] = MODELS[model_name](**config[model_key])

    logger.infov("Model is created.")

    return model