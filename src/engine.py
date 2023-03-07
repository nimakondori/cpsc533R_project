import torch
from src.utils.util import visualize_LVID, visualize_LVOT
from src.builders import  dataloader_builder, dataset_builder
import wandb
from tqdm import tqdm



class BaseEngine(object):

    def __init__(self, config, logger, save_dir):
        # Assign a logger and save dir
        self.logger = logger
        self.save_dir = save_dir

        # Load configurations
        self.data_config = config['data']
        self.train_config = config['train']
        # self.model_config = config['model']
        # self.eval_config = config['eval']

        # Seed for reproducability 
        # seed = self.train_config.get('seed', 200)
        seed = 42

        # Determine which device to use
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)  # set the manual seed for torch
            self.device = torch.device("cuda")
        else:
            torch.manual_seed(seed)  # set the manual seed for torch
            self.device = torch.device("cpu")
        self.logger.info(f"Seed is {seed}")

        if self.device == 'cpu':
            self.logger.warn('GPU is not available.')
        else:
            self.logger.warn('{} GPU(s) is/are available.'.format(
                torch.cuda.device_count()))

    def run(self):
        pass

    def evaluate(self):
        pass


class Engine(BaseEngine):

    def __init__(self, config, logger, save_dir):
        super(Engine, self).__init__(config, logger, save_dir)

    def _build(self, mode='train'):
        # Create datasets
        datasets = dataset_builder.build(data_config=self.data_config, logger=self.logger)

        # Build a dataloader
        self.dataloaders = dataloader_builder.build(datasets=datasets,
                                                    train_config=self.train_config,
                                                    logger=self.logger,
                                                    use_data_parallel=True if torch.cuda.device_count() > 1 else False)

    def run(self):
        # Get needed config here

        self._build(mode='train')
        self.logger.info("datasets successfully built.")
        
        # Do data sanity check here 
        lvid_train_iter = iter(self.dataloaders['lvidlandmark']['train'])
        lvot_train_iter = iter(self.dataloaders['lvotlandmark']['train'])

        LVID_smaple_batch = next(lvid_train_iter)
        LVOT_smaple_batch = next(lvot_train_iter)

        # TODO: Find out why batch size doesn't work as expected
        visualize_LVID(LVID_smaple_batch) 
        visualize_LVOT(LVOT_smaple_batch) 
        

        

    