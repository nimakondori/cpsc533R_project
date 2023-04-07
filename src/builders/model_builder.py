from src.core.models import UMT, CNN_Basic, ViT
from copy import deepcopy


MODELS = {
    'umt': UMT,
    'cnn_basic': CNN_Basic,
    'vit': ViT,
}


def build(config, logger):
    config = deepcopy(config)
    _ = config.pop('checkpoint_path')
    model_name = config.pop('name')

    # Build a model      
    model = MODELS[model_name](**config)
    logger.infov("Model is created.")

    return model