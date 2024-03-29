import logging
import os
import yaml
import argparse
from colorlog import ColoredFormatter
from typing import Dict, Any
from distutils.util import strtobool
import matplotlib.pyplot as plt


def load_log(name):
    def _infov(self, msg, *args, **kwargs):
        self.log(logging.INFO + 1, msg, *args, **kwargs)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = ColoredFormatter(
        "%(log_color)s[%(asctime)s - %(name)s] %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'white,bold',
            'INFOV':    'cyan,bold',
            'WARNING':  'yellow',
            'ERROR':    'red,bold',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    ch.setFormatter(formatter)

    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.handlers = []       # No duplicated handlers
    log.propagate = False   # workaround for duplicated logs in ipython
    log.addHandler(ch)

    logging.addLevelName(logging.INFO + 1, 'INFOV')
    logging.Logger.infov = _infov
    return log


def load_config(config_path) -> dict:
    """
    This functions reads an input config file and returns a dictionary of configurations.
    args:
        config_path (string): path to config file
    returns:
        config (dict)
    """
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def updated_config() -> Dict[str, Any]:
    # creating an initial parser to read the config.yml file.
    initial_parser = argparse.ArgumentParser()
    initial_parser.add_argument('--config_path', default="",
                                help="Path to a config")
    initial_parser.add_argument('--save_dir', default="",
                                help='Path to dir to save train dirs')
    initial_parser.add_argument("--eval_only", type=lambda x: bool(strtobool(x)), default=False,
                                help="evaluate only if it is true")
    initial_parser.add_argument('--eval_data_type', default='val',
                                help='data split for evaluation. either val or test')
    args, unknown = initial_parser.parse_known_args()
    config = load_config(args.config_path)
    config['config_path'] = args.config_path
    config['save_dir'] = args.save_dir
    config['eval_only'] = args.eval_only
    config['eval_data_type'] = args.eval_data_type

    def get_type_v(v):
        """
        for boolean configs, return a lambda type for argparser so string input can be converted to boolean
        """
        if type(v) == bool:
            return lambda x: bool(strtobool(x))
        else:
            return type(v)

    # creating a final parser with arguments relevant to the config.yml file
    parser = argparse.ArgumentParser()
    for k, v in config.items():
        if type(v) is not dict:
            parser.add_argument(f'--{k}', type=get_type_v(v), default=None)
        else:
            for k2, v2 in v.items():
                if type(v2) is not dict:
                    parser.add_argument(f'--{k}.{k2}', type=get_type_v(v2), default=None)
                else:
                    for k3, v3 in v2.items():
                        parser.add_argument(f'--{k}.{k2}.{k3}', type=get_type_v(v3), default=None)
    args, unknown = parser.parse_known_args()

    # Update the configuration with the python input arguments
    for k, v in config.items():
        if type(v) is not dict:
            if args.__dict__[k] is not None:
                config[k] = args.__dict__[k]
        else:
            for k2, v2 in v.items():
                if type(v2) is not dict:
                    if args.__dict__[f'{k}.{k2}'] is not None:
                        config[k][k2] = args.__dict__[f'{k}.{k2}']
                else:
                    for k3, v3 in v2.items():
                        if args.__dict__[f'{k}.{k2}.{k3}'] is not None:
                            config[k][k2][k3] = args.__dict__[f'{k}.{k2}.{k3}']

    return config


def mkdir_p(path):
    os.makedirs(path, exist_ok=True)
    return path


def normalization_params():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return (mean, std)


def to_device(dict, device):
    for key in dict.keys():
        dict[key] = dict[key].to(device)
    return dict


def reset_evaluators(evaluators):
    """
    Calls the reset() method of evaluators in input dict

    :param evaluators: dict, dictionary of evaluators
    """

    for evaluator in evaluators.keys():
        evaluators[evaluator].reset()


def visualize_LVID(image, gt_labels, pred_labels, save_path=None, file_name=None):
    fig = plt.figure()
    axes = [fig.add_subplot(1,2,1), fig.add_subplot(1,2,2)]

    image = image.detach().cpu().numpy()
    gt_labels= gt_labels.detach().cpu().numpy()
    pred_labels= pred_labels.detach().cpu().numpy()

    x = [image, image]
    y = [gt_labels, pred_labels]
    for i in range(len(x)):
        axes[i].imshow(x[i].squeeze().squeeze())
        axes[i].set_title(f"LVID Sample {i+1}")
        axes[i].plot(y[i][0, 1] - 1, y[i][0, 0] - 1, marker='o', color='r', markersize=5)
        axes[i].plot(y[i][1, 1] - 1, y[i][1, 0] - 1, marker='o', color='r', markersize=5)
        axes[i].plot(y[i][2, 1] - 1, y[i][2, 0] - 1, marker='o', color='w', markersize=5)
        axes[i].plot(y[i][3, 1] - 1, y[i][3, 0] - 1, marker='o', color='b', markersize=5)
    
    
    if save_path is not None:        
        file_name = 'sample.png' if file_name is None else file_name + '.png'
        fig.savefig(save_path + file_name)
    else:
        fig.show()
