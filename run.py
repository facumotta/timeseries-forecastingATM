import argparse
import os
import torch
import torch.backends
from exp.forecast import Forecast
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    args = argparse.Namespace(
        task_name='forecast',  # task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]
        is_training=False,  # status
        model_id=f'TimesNet_cajero_2',  # model id
        model='TimesNet',  # model name, options: [Autoformer, Transformer, TimesNet]

        data='ATM',  # dataset type
        root_path='./data_processed/',  # root path of the data file
        data_path=f'df_resultado_cajero_2.csv',  # data file
        features='MS',  # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
        target='TARGET',  # target feature in S or MS task
        freq='d',  # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
        checkpoints='./checkpoints/',  # location of model checkpoints

        seq_len=60,  # input sequence length
        pred_len=1,  # prediction sequence length
        inverse=True,  # inverse output data

        top_k=7,  # for TimesBlock
        num_kernels=5,  # for Inception
        enc_in=9,  # encoder input size
        c_out=1,  # output size
        d_model=512,  # dimension of model
        e_layers=10,  # num of encoder layers
        d_ff=256,  # dimension of fcn
        dropout=0.1,  # dropout
        embed='fixed',  # time features encoding, options:[timeF, fixed, learned]
        activation='gelu',  # activation

        num_workers=10,  # data loader num workers
        itr=1,  # experiments times
        train_epochs=100,  # train epochs
        batch_size=32,  # batch size of train input data
        patience=15,  # early stopping patience
        learning_rate=0.0001,  # optimizer learning rate
        lradj='type1',  # adjust learning rate
        use_amp=False,  # use automatic mixed precision training

        use_gpu=True,  # use gpu
        gpu=0,  # gpu
        gpu_type='cuda',  # gpu type
        use_multi_gpu=False,  # use multiple gpus
        devices='0,1,2,3'  # device ids of multiple gpus
    )


    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.task_name == 'forecast':
        Exp = Forecast
    
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_pl{}_dm{}_el{}_df{}_eb{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.pred_len,
                args.d_model,
                args.e_layers,
                args.d_ff,
                args.embed,
                ii
            )

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_pl{}_dm{}_el{}_df{}_eb{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.e_layers,
            args.d_ff,
            args.embed,
            ii
        )

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
