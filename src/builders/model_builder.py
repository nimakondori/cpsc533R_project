from src.core.models import UMMT, CNN_Basic, ViT, PreTrainedViT
from copy import deepcopy


MODELS = {
    'umtt': UMMT,
    'cnn_basic': CNN_Basic,
    'vit': ViT,
    'pretrained': PreTrainedViT,
}


def build(config, logger):
    config = deepcopy(config)
    _ = config.pop('checkpoint_path')
    model_name = config.pop('name')

    # Build a model      
    model = MODELS[model_name](**config)
    logger.infov("Model is created.")

    return model