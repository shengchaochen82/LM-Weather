import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *
from utils.tools import adjust_learning_rate

class client_LMWeather(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def train(self):
        trainloader = self.load_train_data()


        self.model.to(self.device)
        self.model.train()

        # differential privacy
        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)
        
        start_time = time.time()

        max_local_steps = self.local_epochs
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        for step in range(max_local_steps):
            
            for i, (x, y, x_mark, y_mark) in enumerate(trainloader):

                x = x.float().to(self.device)
                y = y.float().to(self.device)
                x_mark = x_mark.float().to(self.device)
                y_mark = y_mark.float().to(self.device)

                # decoder input
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
                    loss = self.loss(output, y)

                else:
                    if self.args.output_attention:
                        output = self.model(x, x_mark, dec_inp, y_mark)[0]
                    else:
                        output = self.model(x, x_mark, dec_inp, y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    output = output[:, -self.pred_len:, f_dim:]
                    y = y[:, -self.pred_len:, f_dim:].to(self.device)
                    loss = self.loss(output, y)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                self.optimizer.zero_grad()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

            adjust_learning_rate(self.optimizer, step, self.args)
        self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

    def train_imputation(self):
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.train()

        # differential privacy
        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)
        
        start_time = time.time()

        max_local_steps = self.local_epochs
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            
            for i, (x, y, x_mark, y_mark) in enumerate(trainloader):

                x = x.float().to(self.device)
                y = y.float().to(self.device)
                x_mark = x_mark.float().to(self.device)
                y_mark = y_mark.float().to(self.device)

                # Random mask
                B, T, N = x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = x.masked_fill(mask == 0, 0)
                outputs = self.model(inp, x_mark, None, None, mask)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = self.loss(outputs[mask == 0], x[mask == 0])

                # decoder input
                dec_inp = torch.zeros_like(y[:, -self.pred_len:, :]).float()
                dec_inp = torch.cat([y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()

            adjust_learning_rate(self.optimizer, step, self.args)
        self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

    # Personalized for Lora (Global) others local
    def set_parameters(self, model):
        for (nn, np), (on, op) in zip(model.named_parameters(), self.model.named_parameters()):
            # layer contains low-rank parameters
            if 'layer' not in nn:
                op.data = np.data.clone()