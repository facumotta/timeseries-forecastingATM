import os
import time
import numpy as np
import torch
from layers.Embed import DataEmbedding
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from utils.data_loader import Dataset_ATM_day


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience # how many times will you tolerate for loss not being on decrease
        self.verbose = verbose  # whether to print tip info
        self.counter = 0 # now how many times loss not on decrease
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)

        # meaning: current score is not 'delta' better than best_score, representing that 
        # further training may not bring remarkable improvement in loss. 
        elif score < self.best_score + self.delta:  
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            # 'No Improvement' times become higher than patience --> Stop Further Training
            if self.counter >= self.patience:
                self.early_stop = True

        else: #model's loss is still on decrease, save the now best model and go on training
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
    ### used for saving the current best model
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, args):

    #first type: learning rate decrease with epoch by exponential
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}

    #second type: learning rate decrease manually
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }

    #1st type: update in each epoch
    #2nd type: only update in epochs that are written in Dict lr_adjust
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
    
        # change the learning rate for different parameter groups within the optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')



def data_provider(args, flag):
    # time features encoding, options:[timeF, fixed, learned]
    timeenc = 0 if args.embed != 'timeF' else 1

    #test data provider
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        # batch size for test, usually set to 1
        # because we want to predict one sample at a time
        # and we don't need to shuffle the data during evaluation
        batch_size = 1  # bsz=1 for evaluation

        #freq for time features encoding, 
        # options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly,
        #  m:monthly], you can also use more detailed freq like 15min or 3h')
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq


    if args.task_name == 'forecast':
        # Create a dataset for ATM data with daily frequency
        # and the specified features, target, and seasonal patterns.
        data_set = Dataset_ATM_day(
            root_path=args.root_path, #eg.  ./data/ETT/
            data_path=args.data_path, #eg.  ETTh1.csv
            flag=flag,
            size=[args.seq_len, args.pred_len],
            features=args.features,   #forecasting task, options:[M, S, MS]; 
            # M:multivariate predict multivariate, S:univariate predict univariate,
            # MS:multivariate predict univariate
            
            target=args.target,       #target feature in S or MS task
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=None
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=False if flag == 'test' else drop_last)
        
        return data_set, data_loader