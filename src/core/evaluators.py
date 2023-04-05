import torch
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error
import torchvision
import matplotlib.pyplot as plt

# TODO: Fully review this file
class BinaryAccuracyEvaluator(object):

    def __init__(self,  logger):

        self.score = 0.
        self.count = 0

    def reset(self):

        self.score = 0.
        self.count = 0

    def update(self, y_pred, y_true):

        self.count += 1
        self.score += accuracy_score(y_true=y_true, y_pred=y_pred > 0.5)

    def compute(self):

        return self.score / self.count


class MSEEvaluator(object):

    def __init__(self,  logger):

        self.score_per_class = None  # shape b,4

    def reset(self):

        self.score_per_class = None  # shape b,4

    def update(self, y_pred, y_true):
        """
        y_true: true binary labels for each node (num_nodes, num_classes)
        y_pred: sigmoid outputs (num_nodes, num_classes)
        """
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()

        # per class accuracy calculation
        if self.score_per_class is None:
            self.score_per_class = self.compute_per_class(y_pred, y_true)
        else:
            # becomes a numpy array with shape of (b,2)
            self.score_per_class = np.append(self.score_per_class, self.compute_per_class(y_pred, y_true), axis=0)

    def compute(self):
        """
        computes average of MSE across landmarks
        """

        return self.get_per_class_score().mean() # first average across the batch, then across  4 landmark errors

    def get_per_class_score(self):
        """
        returns the average MSEs for all classes
        """

        return self.score_per_class.mean(axis=0)

    def get_last(self):
        return self.score_per_class[-1, :].mean()

    @staticmethod
    def compute_per_class(y_pred, y_true):
        """ computes MSE score for each of the landmarks separately
        score_per_class (array): shape (num_classes,)
        """

        score_per_class = []
        for idx in range(y_true.shape[-1]):
            score_per_class.append(mean_squared_error(y_true=y_true[:, idx], y_pred=y_pred[:, idx]))

        return np.asarray(score_per_class).reshape((1,-1))


class BalancedBinaryAccuracyEvaluator(object):

    def __init__(self,  logger):

        self.score_per_class = None # shape b,4

    def reset(self):

        self.score_per_class = None

    def update(self, y_pred, y_true, valid):
        """
        y_true: true binary labels for each node (num_nodes, num_classes)
        y_pred: sigmoid outputs (num_nodes, num_classes)
        """
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()

        # per class accuracy calculation
        if self.score_per_class is None:
            self.score_per_class = self.compute_per_class(y_pred, y_true, valid)
        else:
            # becomes a numpy array with shape of (b,4)
            self.score_per_class = np.append(self.score_per_class, self.compute_per_class(y_pred,
                                                                                          y_true,
                                                                                          valid), axis=0)

    def compute(self):
        """
        computes average of balanced binary accuracies across landmarks
        """

        return self.score_per_class.mean(axis=0).mean() # first average across the batch, then across  4 landmark errors

    def get_per_class_score(self):
        """
        returns the average accuracy for all classes
        """

        return self.score_per_class.mean(axis=0)

    def get_last(self):
        return self.score_per_class[-1,:].mean()

    @staticmethod
    def compute_per_class(y_pred, y_true, valid):
        """ computes balanced binary accuracy for each of the landmarks separately
        score_per_class (array): shape (num_classes,)
        """

        score_per_class = []
        for idx in range(y_true.shape[-1]):
            if torch.count_nonzero(valid[:, idx]) > 0:
                score_per_class.append(balanced_accuracy_score(y_true=y_true[:, idx][valid[:, idx] > 0],
                                                               y_pred=y_pred[:, idx][valid[:, idx] > 0] > 0.5))
            else:
                score_per_class.append(0)

        return np.asarray(score_per_class).reshape((1, -1))


class LandmarkErrorEvaluator(object):
    def __init__(self, logger, batch_size, frame_size, use_coord_graph):

        self.errors = {'lvid': [],
                       'ivs': [],
                       'lvpw': []}
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.use_coord_graph = use_coord_graph

    def reset(self):

        self.errors = {'lvid': [],
                       'ivs': [],
                       'lvpw': []}

    def update(self, y_pred, y_true):

        nodes_in_batch = y_pred.size(0) // self.batch_size

        y_pred = y_pred.view(self.batch_size, nodes_in_batch, 4).detach().cpu().numpy()
        y_true = y_true.view(self.batch_size, nodes_in_batch, 4).detach().cpu().numpy()

        lvid_err = []
        lvpw_err = []
        ivs_err = []

        for i in range(self.batch_size):
            pred_heatmap = torch.tensor(y_pred[i, -self.frame_size*self.frame_size:, :]).view(self.frame_size,
                                                                                              self.frame_size,
                                                                                              4)
            gt_heatmap = torch.tensor(y_true[i, -self.frame_size*self.frame_size:, :]).view(self.frame_size,
                                                                                            self.frame_size,
                                                                                            4)

            # lvid_top, lvid_bot, lvpw, ivs
            gt_x = torch.argmax(torch.argmax(gt_heatmap, 0), 0)
            gt_y = torch.argmax(torch.argmax(gt_heatmap, 1), 0)
            gt_lvid = self.get_pixel_length(gt_x, gt_y, 0, 1)
            gt_ivs = self.get_pixel_length(gt_x, gt_y, 0, 3)
            gt_lvpw = self.get_pixel_length(gt_x, gt_y, 2, 1)

            preds_x = torch.argmax(torch.argmax(pred_heatmap, 0), 0)
            preds_y = torch.argmax(torch.argmax(pred_heatmap, 1), 0)
            pred_lvid = self.get_pixel_length(preds_x, preds_y, 0, 1)
            pred_ivs = self.get_pixel_length(preds_x, preds_y, 0, 3)
            pred_lvpw = self.get_pixel_length(preds_x, preds_y, 2, 1)

            lvid_err.append(torch.abs(pred_lvid - gt_lvid))
            ivs_err.append(torch.abs(pred_ivs - gt_ivs))
            lvpw_err.append(torch.abs(pred_lvpw - gt_lvpw))

        self.errors['lvid'].append(np.mean(lvid_err))
        self.errors['ivs'].append(np.mean(ivs_err))
        self.errors['lvpw'].append(np.mean(lvpw_err))

    def compute(self):
        """
        compute the mean of all the recorded
        """

        temp = dict()
        temp['ivs_w'] = np.asarray(self.errors['ivs']).mean()
        temp['lvid_w'] = np.asarray(self.errors['lvid']).mean()
        temp['lvpw_w'] = np.asarray(self.errors['lvpw']).mean()

        return temp

    def get_last(self):
        temp = dict()
        temp['ivs_w'] = self.errors['ivs'][-1]
        temp['lvid_w'] = self.errors['lvid'][-1]
        temp['lvpw_w'] = self.errors['lvpw'][-1]

        return temp

    def get_heatmaps(self, y_pred):
        titles = ['lvid_top', 'lvid_bot', 'lvpw', 'ivs']
        nodes_in_batch = y_pred.size(0) // self.batch_size
        y_pred = y_pred.view(self.batch_size, nodes_in_batch, 4).detach().cpu().numpy()
        hms = y_pred[0, -self.frame_size*self.frame_size:, :].reshape(self.frame_size, self.frame_size, 4)
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        for i in range(4):
            axs[i].imshow(hms[:, :, i], cmap='cool')
            axs[i].set_title(titles[i])
            axs[i].axis('off')
        return fig

    @staticmethod
    def get_pixel_length(x, y, i, j):
        return torch.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)


class LandmarkExpectedCoordiantesEvaluator(object):
    """
    to locate the landmarks by finding the expected value of a softmaxed heatmap and calcualte how far they are from gt
    """
    def __init__(self, logger, batch_size, frame_size):

        self.coordinate_errors = {'ivs': [],
                                  'lvid_top': [],
                                  'lvid_bot': [],
                                  'lvpw': []}
        self.width_MAE = {'lvid': [],
                          'ivs': [],
                          'lvpw': []}
        self.width_MPE = {'lvid': [],
                          'ivs': [],
                          'lvpw': []}
        self.detailed_performance = {}
        self.batch_size = batch_size
        self.frame_size = frame_size
    def reset(self):

        self.coordinate_errors = {'ivs': [],
                                  'lvid_top': [],
                                  'lvid_bot': [],
                                  'lvpw': []}
        self.width_MAE = {'lvid': [],
                          'ivs': [],
                          'lvpw': []}
        self.width_MPE = {'lvid': [],
                          'ivs': [],
                          'lvpw': []}
        self.detailed_performance.clear()

    @staticmethod
    def initialize_maps(length, shape):
        '''
            creates an initial map for x and y elements with shape (1, length, 1, 1) or (1, 1, length, 1)
        '''
        line = torch.linspace(0, length - 1, length)
        map_init = torch.reshape(line, shape)
        return map_init

    def update(self, y_pred, y_true, pix2mm_x, pix2mm_y):
        """
        shapes are (batch_size x nodes_in_batch, num_landmarks) = (b x nodes_in_batch, 4)
        y_pred: tensor of logits for node prediction before softmax.
        y_true: tensor of node ground truth
        """
        self.detailed_performance.clear()


        gt_w, gt_h = y_true[:, :, 0], y_true[:, :, 1]      
        preds_w, preds_h = y_pred[:, :, 0], y_pred[:, :, 1]      

        # calculating error between pred/gt coordinates
        landmark_errors = self.get_pixel_length(gt_w, gt_h,
                                                preds_w, preds_h,
                                                pix2mm_x.unsqueeze(1),
                                                pix2mm_y.unsqueeze(1)).numpy()  # shape (b, 4)

        landmark_errors = np.squeeze(np.sum(landmark_errors, axis=1) / self.batch_size) # size (4)

        self.coordinate_errors['lvid_top'].append(landmark_errors[0])
        self.coordinate_errors['lvid_bot'].append(landmark_errors[1])
        self.coordinate_errors['lvpw'].append(landmark_errors[2])
        self.coordinate_errors['ivs'].append(landmark_errors[3])

        # calculating error between pred/gt widths
        widths = self.calculate_widths(y_pred, y_true, pix2mm_x, pix2mm_y)  # shape (b)
        ivs_err, lvid_err, lvpw_err = self.calculate_width_MAE(widths)  # shape (b)

        self.width_MAE['ivs'].append(ivs_err.sum().item() / self.batch_size)    # Average per batch
        self.width_MAE['lvid'].append(lvid_err.sum().item() / self.batch_size)  # Average per batch
        self.width_MAE['lvpw'].append(lvpw_err.sum().item() / self.batch_size)  # Average per batch

        ivs_err, lvid_err, lvpw_err = self.calculate_width_MPE(widths)  # shape (b)

        self.width_MPE['ivs'].append(ivs_err.sum().item() / self.batch_size)    # Average per batch
        self.width_MPE['lvid'].append(lvid_err.sum().item() / self.batch_size)  # Average per batch
        self.width_MPE['lvpw'].append(lvpw_err.sum().item() / self.batch_size)  # Average per batch


        # recording the performance to be accessed later
        coordinates = {
            'pred_ivs': y_pred[:,3], 'pred_lvid_top': y_pred[:,0], 'pred_lvid_bot': y_pred[:,1], 'pred_lvpw': y_pred[:,2],
            'gt_ivs': y_true[:, 3], 'gt_lvid_top': y_true[:, 0], 'gt_lvid_bot': y_true[:, 1], 'gt_lvpw': y_true[:, 2]
        }
        self.detailed_performance = {'widths': widths, 'coordinates': coordinates}

    def calculate_widths(self, preds, gt, pix2mm_x, pix2mm_y):
        """
        input shapes are b,4,2
        output values of the dictionary are tensors with shape of (b)
        """

        widths = {"pred_ivs_mm": self.get_pixel_length(preds[:,3,1], preds[:,3,0], preds[:,0,1], preds[:,0,0], pix2mm_x, pix2mm_y),
                  "pred_lvid_mm": self.get_pixel_length(preds[:,0,1], preds[:,0,0], preds[:,1,1], preds[:,1,0], pix2mm_x, pix2mm_y),
                  "pred_lvpw_mm": self.get_pixel_length(preds[:, 1, 1], preds[:, 1, 0], preds[:, 2, 1], preds[:, 2, 0], pix2mm_x, pix2mm_y),
                  "gt_ivs_mm": self.get_pixel_length(gt[:,3,1], gt[:,3,0], gt[:,0,1], gt[:,0,0], pix2mm_x, pix2mm_y),
                  "gt_lvid_mm": self.get_pixel_length(gt[:, 0, 1], gt[:, 0, 0], gt[:, 1, 1], gt[:, 1, 0], pix2mm_x,pix2mm_y),
                  "gt_lvpw_mm": self.get_pixel_length(gt[:, 1, 1], gt[:, 1, 0], gt[:, 2, 1], gt[:, 2, 0], pix2mm_x,pix2mm_y)}

        return widths

    def calculate_width_MAE(self, widths):
        """
        calculate the absolute errors between predicted and gt width of the landmarks. shape (b)
        """
        ivs_err = torch.abs(widths['pred_ivs_mm'] - widths['gt_ivs_mm'])
        lvid_err = torch.abs(widths['pred_lvid_mm'] - widths['gt_lvid_mm'])
        lvpw_err = torch.abs(widths['pred_lvpw_mm'] - widths['gt_lvpw_mm'])
        return ivs_err, lvid_err, lvpw_err

    def calculate_width_MPE(self, widths):
        """
        calculate the percentage errors between predicted and gt width of the landmarks. shape (b)
        """
        ivs_err = 100 * torch.abs(widths['pred_ivs_mm'] - widths['gt_ivs_mm']) / widths['gt_ivs_mm']
        lvid_err = 100 * torch.abs(widths['pred_lvid_mm'] - widths['gt_lvid_mm']) / widths['gt_lvid_mm']
        lvpw_err = 100 * torch.abs(widths['pred_lvpw_mm'] - widths['gt_lvpw_mm']) / widths['gt_lvpw_mm']
        return ivs_err, lvid_err, lvpw_err

    def compute(self):
        """
        compute the mean of all the recorded
        """

        temp = dict()
        temp['lvid_top'] = np.asarray(self.coordinate_errors['lvid_top']).mean()
        temp['lvid_bot'] = np.asarray(self.coordinate_errors['lvid_bot']).mean()
        temp['lvpw'] = np.asarray(self.coordinate_errors['lvpw']).mean()
        temp['ivs'] = np.asarray(self.coordinate_errors['ivs']).mean()

        temp['ivs_w'] = np.asarray(self.width_MAE['ivs']).mean()
        temp['lvid_w'] = np.asarray(self.width_MAE['lvid']).mean()
        temp['lvpw_w'] = np.asarray(self.width_MAE['lvpw']).mean()

        temp['ivs_mpe'] = np.asarray(self.width_MPE['ivs']).mean()
        temp['lvid_mpe'] = np.asarray(self.width_MPE['lvid']).mean()
        temp['lvpw_mpe'] = np.asarray(self.width_MPE['lvpw']).mean()

        return temp

    def get_sum_of_width_MAE(self):
        """
        returns the sum of widths mean absolute errors
        """
        temp = self.compute()
        return sum([value for k, value in temp.items() if k in ['ivs_w', 'lvid_w', 'lvpw_w']])

    def get_sum_of_width_MPE(self):
        """
        returns the sum of widths' mean percent errors
        """
        temp = self.compute()
        return sum([value for k, value in temp.items() if k in ['ivs_mpe', 'lvid_mpe', 'lvpw_mpe']])

    def get_last(self):
        temp = dict()
        temp['lvid_top'] = self.coordinate_errors['lvid_top'][-1]
        temp['lvid_bot'] = self.coordinate_errors['lvid_bot'][-1]
        temp['lvpw'] = self.coordinate_errors['lvpw'][-1]
        temp['ivs'] = self.coordinate_errors['ivs'][-1]

        temp['ivs_w'] = self.width_MAE['ivs'][-1]
        temp['lvid_w'] = self.width_MAE['lvid'][-1]
        temp['lvpw_w'] = self.width_MAE['lvpw'][-1]

        temp['ivs_mpe'] = self.width_MPE['ivs'][-1]
        temp['lvid_mpe'] = self.width_MPE['lvid'][-1]
        temp['lvpw_mpe'] = self.width_MPE['lvpw'][-1]

        return temp

    def get_predictions(self):
        """
        returns a dictionary of lists containing detailed performance of model in the current iteration.
        each value is in list format with length of batch_size
        """
        return self.detailed_performance



    def create_overlay_image(self, x, hms):
        """
        x is grayscale image. tensor with shape (H,W)
        hms is the matrix of heatmaps. tensor with shape (H,W,Channels)

        returns a PIL Image object
        """
        cmap = plt.get_cmap('hsv')
        img = torch.zeros(3, self.frame_size, self.frame_size)
        # for grayscale
        color = torch.FloatTensor((0.8,.8,.8)).reshape([-1, 1, 1])
        img = torch.max(img, color * x)  # max in case of overlapping position of joints
        # for heatmaps
        colors_rgb = [(0,1,1), (1,0.7,0.9), (0,1,0), (1,0,0)]
        # C = 4
        for i, color_rgb in enumerate(colors_rgb):
            # color = torch.FloatTensor(cmap(i * cmap.N // C)[:3]).reshape([-1, 1, 1])
            color = torch.FloatTensor(color_rgb).reshape([-1, 1, 1])
            img = torch.max(img, color * hms[:,:,i])  # max in case of overlapping position of joints
        img = torchvision.transforms.ToPILImage()(img)        # fig, axs = plt.subplots(1, 4, figsize=(16, 4))

        return img

    @staticmethod
    def get_pixel_length(x0, y0, x1, y1, pix2mm_x, pix2mm_y):
        return torch.sqrt(((x0 - x1)*pix2mm_x) ** 2 + ((y0 - y1)*pix2mm_y) ** 2)