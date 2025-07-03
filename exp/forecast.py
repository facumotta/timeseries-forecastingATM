from exp.set_up import Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, data_provider
from utils.metrics import MAE, MSE
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings('ignore')


class Forecast(Basic):
    def __init__(self, args):
        super(Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        # if self.args.use_multi_gpu and self.args.use_gpu:
        #     model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion


    def train(self, setting):  #setting is the args for this model training
        #get train dataloader
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')    

        # set path of checkpoint for saving and loading model
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()

        train_steps = len(train_loader)

        # EarlyStopping is typically a custom class or function that monitors the performance 
        # of a model during training, usually by tracking a certain metric (commonly validation 
        # loss or accuracy).It's a common technique used in deep learning to prevent overfitting 
        # during the training
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        #Optimizer and Loss Function Selection
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # AMP training is a technique that uses lower-precision data types (e.g., float16) 
        # for certain computations to accelerate training and reduce memory usage.
        if self.args.use_amp:  
            scaler = torch.cuda.amp.GradScaler()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            #begin training in this epoch
            for i, (batch_x, batch_y, batch_x_mark, _) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)  #input features
                batch_y = batch_y.float().to(self.device)  #target features

                # _mark holds information about time-related features. Specifically, it is a 
                # tensor that encodes temporal information and is associated with the 
                # input data batch_x.
                batch_x_mark = batch_x_mark.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp: #in the case of TimesNet, use_amp should be False
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark)

                        # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, 
                        # S:univariate predict univariate, MS:multivariate predict univariate'
                        #if multivariate predict univariate',then output should be the last column of the decoder
                        # output, so f_dim = -1 to only contain the last column, else is all columns
                        f_dim = -1 if self.args.features == 'MS' else 0 
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        # calc loss
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:  #similar to when use_amp is True
                    outputs = self.model(batch_x, batch_x_mark)
                    
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                # When train rounds attain some 5-multiple, print speed, left time, loss. etc feedback
                if (i + 1) % 5 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                #BP
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            
            #This epoch comes to end, print information
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            #run test and validation on current model
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            #print train, test, vali loss information
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            #Decide whether to trigger Early Stopping. if early_stop is true, it means that 
            #this epoch's training is now at a flat slope, so stop further training for this epoch.
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            #adjust learning keys
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        best_model_path = path + '/' + 'checkpoint.pth'

        # loading the trained model's state dictionary from a saved checkpoint file 
        # located at best_model_path.
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model



    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []

        #evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    outputs = self.model(batch_x, batch_x_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss



    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        y_marks = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                #inverse the data if scaled
                if test_data.scale and self.args.inverse:
                    shape_o = outputs.shape  # [B, pred_len, 1]
                    shape_y = batch_y.shape  # [B, pred_len, 1]

                    # Extraemos solo TARGET (ya es la única columna en outputs)
                    outputs_2d = outputs.reshape(-1, 1)   # [B * pred_len, 1]
                    batch_y_2d = batch_y.reshape(-1, 1)

                    # Invertimos con el target_scaler entrenado solo con 'TARGET'
                    outputs_2d = test_data.target_scaler.inverse_transform(outputs_2d)
                    batch_y_2d = test_data.target_scaler.inverse_transform(batch_y_2d)

                    # Volvemos a la forma original
                    outputs = outputs_2d.reshape(shape_o)
                    batch_y = batch_y_2d.reshape(shape_y)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                y_marks.append(batch_y_mark.detach().cpu().numpy())

                #visualize one piece of data every 20
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    #the whole sequence
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)  # shape[batch_num, batch_size, pred_len, features]
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        y_marks = np.array(y_marks)
        y_marks = y_marks.reshape(-1, y_marks.shape[-2], y_marks.shape[-1])
        print('test shape:', preds.shape, trues.shape, y_marks.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae = MAE(preds, trues)
        mse = MSE(preds, trues)
        
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()
        
        np.save(folder_path + 'metrics.npy', np.array([mae, mse]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'y_marks.npy', y_marks)

        return
