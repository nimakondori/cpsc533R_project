from datetime import timedelta
import pandas as pd
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import os.path as osp

from src.utils import util
from src.utils.util import visualize_LVID
from src.builders import  dataloader_builder, dataset_builder, model_builder, optimizer_builder, \
                            scheduler_builder, criterion_builder, evaluator_builder, meter_builder, \
                            checkpointer_builder
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
        self.model_config = config['model']
        self.eval_config = config['eval']

        # Seed for reproducability 
        seed = self.train_config['seed']
        self.logger.info(f"Seed is {seed}")
        
        # Determine which device to use
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)  # set the manual seed for torch
            self.device = torch.device("cuda")
        else:
            torch.manual_seed(seed)  # set the manual seed for torch
            self.device = torch.device("cpu")

        if self.device == 'cpu':
            self.logger.warn('GPU is not available.')
        else:
            self.logger.warn('{} GPU(s) is/are available.'.format(
                torch.cuda.device_count()))
         # Set up Wandb if required
        if config['train']['use_wandb']:
            wandb.init(project=config['train']['wand_project_name'],
                       name=None if config['train']['wandb_run_name'] == '' else config['train']['wandb_run_name'],
                       config=config,
                       mode=config['train']['wandb_mode'])

            # define our custom x axis metric
            wandb.define_metric("batch_train/step")
            wandb.define_metric("batch_valid/step")
            wandb.define_metric("epoch")
            # set all other metrics to use the corresponding step
            wandb.define_metric("batch_train/*", step_metric="batch_train/step")
            wandb.define_metric("batch_valid/*", step_metric="batch_valid/step")
            wandb.define_metric("epoch/*", step_metric="epoch")
            wandb.define_metric("lr", step_metric="epoch")                    
        self.wandb_log_steps = config['train'].get('wandb_log_steps', 1000)

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

        # Build the model
        self.num_output_channels = 2
        self.model = model_builder.build(
            config = self.model_config, logger = self.logger)
                
        # Use multi GPUs if available
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Update the model devices        
        self.model = self.model.to(self.device)

        # Build the optimizer
        self.optimizer = optimizer_builder.build(
            config = self.train_config["optimizer"], model = self.model, logger = self.logger)
        
        # Build the scheduler
        self.scheduler = scheduler_builder.build(
            config = self.train_config, optimizer=self.optimizer, logger = self.logger)

        # Build the criterion
        self.criterion = criterion_builder.build(
            config = self.train_config['criterion'], logger = self.logger)

        # add other required configs
        # TODO: Currently hardcoding the frame size and batch size from LVID stuff, need to fix this
        self.eval_config.update({'frame_size': self.data_config[0]['dataset']['transform']['image_size'],
                                 'batch_size': self.train_config['batch_size']})
       
        # Build the loss meter
        self.loss_meter = meter_builder.build(self.logger)

        # Build the evaluator
        self.evaluators = evaluator_builder.build(
            config = self.eval_config, logger = self.logger) 

        # Build the checkpointer
        self.checkpointer = checkpointer_builder.build(
            self.save_dir, self.logger, self.model, self.optimizer,
            self.scheduler, self.eval_config['standard'], self.eval_config['best_mode'])        
        
        # Load the checkpoint
        checkpoint_path = self.model_config.get('checkpoint_path', '')
        self.misc = self.checkpointer.load(mode, checkpoint_path, use_latest=False)                                       


    def run(self):
        # Get needed config here
        start_epoch, num_steps = 0, 0
        num_epochs = self.train_config.get('num_epochs', 500)
        checkpoint_step = self.train_config.get('checkpoint_step', 1000)

        self._build(mode='train')
        self.logger.info("datasets successfully built.")                                    
        self.logger.info(
            'Train for {} epochs starting from epoch {}'.format(num_epochs, start_epoch))

        for epoch in range(start_epoch, start_epoch + num_epochs):
            util.reset_evaluators(self.evaluators)
            self.loss_meter.reset()

            train_start = time.time()
            num_steps = self._train_one_epoch(epoch, num_steps, checkpoint_step)
            train_time = time.time() - train_start

            # print a summary of the training epoch            
            self.log_wandb({'loss_total': self.loss_meter.avg}, {"epoch": epoch}, mode='epoch/train')
            self.log_summary("Training", epoch, train_time)

            if self.train_config['lr_schedule']['name'] == 'multi':
                self.scheduler.step()
            self.loss_meter.reset()
            util.reset_evaluators(self.evaluators)

            # Evaluate            
            train_start = time.time()
            self._evaluate_once(epoch, num_steps)
            validation_time = time.time() - train_start

            # step lr scheduler with the sum of landmark width errors
            # if self.train_config['lr_schedule']['name'] == 'reduce_lr_on_plateau':
            #     self.scheduler.step(self.evaluators["landmarkcoorderror"].get_sum_of_width_MAE())

            # self.checkpointer.save(epoch,
            #                        num_steps,
            #                        self.evaluators["landmarkcoorderror"].get_sum_of_width_MPE(),
            #                        best_mode='min')
            if epoch % 10 == 0:
                self.checkpointer.save(epoch, num_steps)
            
            self.log_wandb({'loss_total': self.loss_meter.avg}, {"epoch": epoch}, mode='epoch/valid')            
            self.log_summary("Validation", epoch, validation_time)

    def _train_one_epoch(self, epoch, num_steps, checkpoint_step):                
        dataloader = self.dataloaders['lvidlandmark']['train']                
        self.model.train()        
        epoch_steps = len(dataloader)
        data_iter = iter(dataloader)              
        iterator = tqdm(range(epoch_steps), dynamic_ncols=True)
        for i in iterator:                        
            data_batch = next(data_iter)                        
            data_batch = self.set_device(data_batch, self.device)        
            landmark_preds = self.model(data_batch["x"])                          
            losses = self.compute_loss(landmark_preds=landmark_preds, landmark_y=data_batch['y'])                        
            loss = sum(losses.values())            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                batch_size = data_batch['x'].shape[0]                
                self.loss_meter.update(loss.item(), batch_size)

                # update evaluators
                self.update_evaluators(landmark_preds=landmark_preds, landmark_y=data_batch['y'])

                # update tqdm progress bar
                self.set_tqdm_description(iterator, 'train', epoch, loss.item())

                if self.train_config['use_wandb']:
                    step = (epoch*epoch_steps + i)*batch_size
                    self.log_wandb(losses, {"step":step}, mode='batch_train')
                
                num_steps += batch_size                                    

        torch.cuda.empty_cache()
        return num_steps

    def evaluate(self, data_type='val'):
        num_steps = 0
        self._build(mode='test')
        self.logger.info('Evaluating the model')
        util.reset_evaluators(self.evaluators)
        self.loss_meter.reset()

        # Evaluate
        train_start = time.time()
        self._evaluate_once(0, num_steps, data_type=data_type, save_output=True)
        validation_time = time.time() - train_start        
        self.log_summary("Validation", 0, validation_time)
        self.log_wandb({'loss_total': self.loss_meter.avg}, {"epoch": 0}, mode='epoch/valid')

    def _evaluate_once(self, epoch, num_steps, data_type='val', save_output=False):
        # for generating a csv output of model's prediction on dataset
        if save_output: 
            prediction_df = pd.DataFrame()

        dataloader = self.dataloaders['lvidlandmark'][data_type]                
        self.model.eval()

        epoch_steps = len(dataloader)
        data_iter = iter(dataloader)
        iterator = tqdm(range(len(dataloader)), dynamic_ncols=True)
        for i in iterator:
            data_batch = next(data_iter)
            with torch.no_grad():
                data_batch = self.set_device(data_batch, self.device)            
                landmark_preds= self.model(data_batch["x"])
                losses = self.compute_loss(landmark_preds, data_batch["y"])                
                loss = sum(losses.values())                                
                batch_size = data_batch['x'].shape[0]
                self.loss_meter.update(loss.item(), batch_size)                

                # update evaluators                
                self.update_evaluators(landmark_preds=landmark_preds, landmark_y=data_batch['y'])

                # update tqdm progress bar
                self.set_tqdm_description(iterator, 'validation', epoch, loss.item())

                if self.train_config['use_wandb']:
                    step = (epoch*epoch_steps + i)*batch_size
                    self.log_wandb(losses, {"step":step}, mode='batch_valid')
                    # plot the heatmaps                                                                                    
                    if num_steps % self.wandb_log_steps == 0:
                        self.log_heatmap_wandb({"step": step},
                                               data_batch["x"],
                                               landmark_preds,
                                               data_batch["y"],
                                               landmark_preds,
                                               data_batch["pix2mm_x"],
                                               data_batch["pix2mm_y"],
                                               mode='batch_valid')                          
                
                num_steps += batch_size

                # creating the prediction log table for wandb
                if save_output:
                    prediction_df = pd.concat([prediction_df,  self.create_prediction_df(data_batch)], axis=0)

        if save_output:
            # Prediction Table
            if self.train_config['use_wandb']:
                prediction_log_table = wandb.Table(dataframe=prediction_df)
                wandb.log({f"model_output_{data_type}_dataset": prediction_log_table})
            csv_destination = osp.join(osp.dirname(self.model_config['checkpoint_path']),
                                       f'{data_type}_' +
                                       osp.basename(self.model_config['checkpoint_path'])[:-4] +'.csv')
            prediction_df.to_csv(csv_destination)

        torch.cuda.empty_cache()
        return

    def update_evaluators(self,
                          landmark_preds,
                          landmark_y,
                          coord_preds=None,
                          coord_y=None,
                          pix2mm_x=None,
                          pix2mm_y=None):
        """
        update the evaluators with predictions of the current batch. inputs are in cuda
        """
        landmark_preds, landmark_y = landmark_preds.detach().cpu(), landmark_y.detach().cpu()

        for metric in self.eval_config["standards"]:
            if metric == 'landmarkcoorderror':                
                self.evaluators[metric].update(landmark_preds, landmark_y, pix2mm_x, pix2mm_y)
            else:
                self.evaluators[metric].update(landmark_preds, landmark_y)

    def set_tqdm_description(self, iterator, mode, epoch, loss):
        standard_name = self.eval_config["standard"]
        standard_value = self.evaluators[standard_name].get_last()                
        iterator.set_description("[Epoch {}] | {} | Loss: {:.4f} | "
                                 "{}: {:.4f} | ".format(epoch, mode, loss, standard_name, standard_value),
                                 refresh=True)
    
    def log_summary(self, mode, epoch, time):
        """
        log summary after a full training or validation epoch
        """
        standard_name = self.eval_config["standard"]
        standard_value = self.evaluators[standard_name].compute()
        # errors = self.evaluators['landmarkcoorderror'].compute()
        self.logger.infov(f'{mode} [Epoch {epoch}] with lr: {self.optimizer.param_groups[0]["lr"]:.7} '
                          f'completed in {str(timedelta(seconds=time)):.7} - '
                          f'loss: {self.loss_meter.avg:.4f} - '
                          f'{standard_name}: {standard_value:.2%} - ')
                        #   'errors [IVS, LVID_TOP, LVID_BOT, LVPW] ='
                        #   "[{ivs:.4f}, {lvid_top:.4f}, {lvid_bot:.4f}, {lvpw:.4f}] | "
                        #   "[IVS, LVID, LVPW]: "
                        #   "_MAE_[{ivs_w:.4f}, {lvid_w:.4f}, {lvpw_w:.4f}] "
                        #   "_MPE_[{ivs_mpe:.4f}, {lvid_mpe:.4f}, {lvpw_mpe:.4f}]" .format(**errors))

    def log_wandb(self, losses, step_metric, mode='batch_train'):

        if not self.train_config['use_wandb']:
            return

        step_name, step_value = step_metric.popitem()
        standard_name = self.eval_config["standard"]
        if "batch" in mode:
            standard_value = self.evaluators[standard_name].get_last()
            errors = self.evaluators['landmarkcoorderror'].get_last()
            log_dict = {f'{mode}/{step_name}': step_value}
        elif "epoch" in mode:
            standard_value = self.evaluators[standard_name].compute()
            errors = self.evaluators['landmarkcoorderror'].compute()
            log_dict = {f'{step_name}': step_value,   # both train and valid x axis are called epoch
                        'lr': self.optimizer.param_groups[0]['lr']}  # record the Learning Rate
        else:
            raise("invalid mode for wandb logging")

        log_dict.update({f'{mode}/{standard_name}': standard_value,
                         f'{mode}/lvid_top_error': errors['lvid_top'],
                         f'{mode}/lvid_bot_error': errors['lvid_bot'],
                         f'{mode}/lvpw_error': errors['lvpw'],
                         f'{mode}/ivs_error': errors['ivs'],
                         f'{mode}/lvid_w_error': errors['lvid_w'],
                         f'{mode}/lvpw_w_error': errors['lvpw_w'],
                         f'{mode}/ivs_w_error': errors['ivs_w'],
                         f'{mode}/lvid_w_mpe': errors['lvid_mpe'],
                         f'{mode}/lvpw_w_mpe': errors['lvpw_mpe'],
                         f'{mode}/ivs_w_mpe': errors['ivs_mpe'],})

        for loss_name, loss in losses.items():
            loss = loss.item() if type(loss) == torch.Tensor else loss
            log_dict.update({f'{mode}/{loss_name}': loss})

        wandb.log(log_dict)

    def log_heatmap_wandb(self, step_metric,
                          x, landmark_preds, landmark_y,
                          coord_preds, pix2mm_x, pix2mm_y,
                          mode='batch_train'):
        step_name, step_value = step_metric.popitem()

        landmark_preds, landmark_y = landmark_preds.detach().cpu(), landmark_y.detach().cpu()
        pix2mm_x, pix2mm_y = pix2mm_x.detach().cpu(), pix2mm_y.detach().cpu()

        if self.use_coordinate_graph:
            coord_preds = coord_preds.detach().cpu()

        fig = self.evaluators['landmarkcoorderror'].get_heatmaps(x.detach().cpu(),
                                                                 landmark_preds, landmark_y,
                                                                 coord_preds, pix2mm_x, pix2mm_y)
        wandb.log({f'{mode}/heatmaps': fig,
                   f'{mode}/{step_name}': step_value})
        plt.close()

    def set_device(self, data, device):
        if type(data) == list:
            data = [self.set_device(item, device) for item in data]
        elif type(data) == dict:
            for k, v in data.items():
                data[k] = self.set_device(v, device)
        elif type(data) == str:
            pass
        else: 
            data = data.to(device)
        return data
    
    
    def compute_loss(self, landmark_preds, landmark_y):
        """
        computes and sums all the loss values to be a single number, ready for backpropagation
        """

        losses = dict()        
        landmark_preds = landmark_preds.view(self.train_config['batch_size'], -1, 2*self.num_output_channels)
        landmark_y = landmark_y.view(self.train_config['batch_size'], -1, 2*self.num_output_channels)        
        for criterion_name in self.criterion.keys():
            losses[criterion_name] = self.criterion[criterion_name].compute(landmark_preds, landmark_y)

        return losses
    