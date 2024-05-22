import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics
from utils.data_utils import read_client_data
from utils.metrics import metric
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.tools import visual
from utils.param_utils import calculate_model_size

class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.args = args
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        self.setting = 'task{}_{}_{}_{}_bm{}_sl{}_ll{}_pl{}_fea{}_tag{}_ispeft{}_peft{}_rk{}_pf{}'.format(
                args.task_name,
                args.goal,
                args.algorithm,
                args.dataset,
                args.base_model,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.features,
                args.target,
                args.is_peft,
                args.peft,
                args.rank,
                args.freeze_part)

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay
        self.params_sum = calculate_model_size

        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.label_len = args.label_len

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.args, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.args, self.id, is_train=False)
        return test_data, DataLoader(test_data, batch_size, drop_last=True, shuffle=False)
        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        test_dataset, testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        self.model.to(self.device)
        self.model.eval()

        test_num = 0
        preds = []
        trues = []

        folder_path = './test_results/' + self.setting + '/' + str(self.id) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with torch.no_grad():
            for i, (x, y, x_mark, y_mark) in enumerate(testloaderfull):

                x = x.float().to(self.device)
                y = y.float()

                x_mark = x_mark.float().to(self.device)
                y_mark = y_mark.float().to(self.device)

                # decoder input if exits
                dec_inp = torch.zeros_like(y[:, -self.pred_len:, :]).float()
                dec_inp = torch.cat([y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            output = self.model(x, x_mark, dec_inp, y_mark)[0]
                        else:
                            output = self.model(x, x_mark, dec_inp, y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        output = output[:, -self.pred_len:, f_dim:]
                        y = y[:, -self.pred_len:, f_dim:].to(self.device)
                else:
                    if self.args.output_attention:
                        output = self.model(x, x_mark, dec_inp, y_mark)[0]
                    else:
                        output = self.model(x, x_mark, dec_inp, y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        output = output[:, -self.pred_len:, f_dim:]
                        y = y[:, -self.pred_len:, f_dim:].to(self.device)
                
                pred = output.detach().cpu().numpy()
                true = y.detach().cpu().numpy()

                if test_dataset.scale and self.args.inverse:
                    shape = pred.shape
                    # pred = test_dataset.inverse_transform(pred.reshape(-1, 20)).reshape(shape)
                    # true = test_dataset.inverse_transform(true.reshape(-1, 20)).reshape(shape)
                    pred = test_dataset.inverse_transform(pred.squeeze(0)).reshape(shape)
                    true = test_dataset.inverse_transform(true.squeeze(0)).reshape(shape)

                preds.append(pred)
                trues.append(true)
            
                if i % 20 == 0:
                    input = x.detach().cpu().numpy()
                    if test_dataset.scale and self.args.inverse:
                        shape = input.shape
                        input = test_dataset.inverse_transform(input.reshape(-1, 20)).reshape(shape)
                        input = test_dataset.inverse_transform(input.squeeze(0)).reshape(shape)

                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + "_Client_" + str(self.id) + '.pdf'))

                test_num += true.shape[0]
        
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe, smape, nd = metric(preds, trues)
        print('Client: {}, mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}'.format(self.id, mae * 100, mse * 100, rmse * 100))
        self.model.cpu()

        # results save
        folder_path = './series_results/' + self.setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        f = open("result_long_term_forecast.txt", 'a')
        f.write(self.setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        # self.save_model(self.model, 'model')

        return mae, rmse, mape, test_num
    
    def test_metrics_imputation(self):
        test_dataset, testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        self.model.to(self.device)
        self.model.eval()

        test_num = 0
        preds = []
        trues = []
        masks = []

        folder_path = './test_results/' + self.setting + '/' + str(self.id) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with torch.no_grad():
            for i, (x, y, x_mark, y_mark) in enumerate(testloaderfull):

                x = x.float().to(self.device)
                x_mark = x_mark.float().to(self.device)

                # random mask
                B, T, N = x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = x.masked_fill(mask == 0, 0)

                # imputation
                outputs = self.model(inp, x_mark, None, None, mask)

                # eval
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                true = x.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                masks.append(mask.detach().cpu())
                
                if i % 20 == 0:
                    filled = true[0, :, -1].copy()
                    filled = filled * mask[0, :, -1].detach().cpu().numpy() + \
                             pred[0, :, -1] * (1 - mask[0, :, -1].detach().cpu().numpy())
                    visual(true[0, :, -1], filled, os.path.join(folder_path, str(i) + "_Client_" + str(self.id) + '.pdf'))

                test_num += true.shape[0]
        
        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)
        print('test shape:', preds.shape, trues.shape)
        # Imputation Issues: test shape: (2816, 192, 1) (2816, 192, 20)
        # Note that need to revise for multivariate time series
        mae, mse, rmse, mape, mspe, _, _ = metric(preds[masks == 0], trues[masks == 0])
        print('Client: {}, mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}'.format(self.id, mae * 100, mse * 100, rmse * 100))
        self.model.cpu()

        # results save
        folder_path = './series_results/' + self.setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        f = open("result_long_term_forecast.txt", 'a')
        f.write(self.setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        # self.save_model(self.model, 'model')

        return mae, rmse, mape, test_num

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y, x_mark, y_mark in trainloader:

                x = x.float().to(self.device)
                y = y.float().to(self.device)

                x_mark = x_mark.float().to(self.device)
                y_mark = y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            output = self.model(x, x_mark, dec_inp, y_mark)[0]
                        else:
                            output = self.model(x, x_mark, dec_inp, y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        output = output[:, -self.pred_len:, f_dim:]
                        y = y[:, -self.pred_len:, f_dim:].to(self.device)

                        loss = self.loss(output, y)
                        train_num += y.shape[0]
                        losses += loss.item()
                else:
                    if self.args.output_attention:
                        output = self.model(x, x_mark, dec_inp, y_mark)[0]
                    else:
                        output = self.model(x, x_mark, dec_inp, y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    output = output[:, -self.pred_len:, f_dim:]
                    y = y[:, -self.pred_len:, f_dim:].to(self.device)

                    loss = self.loss(output, y)
                    train_num += y.shape[0]
                    losses += loss.item()

        self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num
    
    def train_metrics_imputation(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y, x_mark, y_mark in trainloader:

                x = x.float().to(self.device)
                y = y.float().to(self.device)

                x_mark = x_mark.float().to(self.device)
                y_mark = y_mark.float().to(self.device)

                # random mask
                B, T, N = x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = x.masked_fill(mask == 0, 0)

                outputs = self.model(inp, x_mark, None, None, mask)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.pred_len:, f_dim:]
                loss = self.loss(outputs[mask == 0], x[mask == 0])

                train_num += y.shape[0]
                losses += loss.item()

        self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))
    
    def check_inf_nan(self, true, pred):
        nan_indices = torch.isnan(true)
        if torch.any(nan_indices):
            true[nan_indices] = 0
        nan_indices1 = torch.isnan(pred)
        if torch.any(nan_indices1):
            pred[nan_indices1] =0
        
        return true, pred

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
