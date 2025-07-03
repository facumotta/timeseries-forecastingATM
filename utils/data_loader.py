import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


class Dataset_ATM_day(Dataset):
    
    def __init__(self, root_path, flag='train', size=None,
                features='MS', data_path='df_resultado_cajero_0.csv',
                target='TARGET', scale=True, timeenc=0, freq='d', seasonal_patterns=None):
        # size [seq_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = 0
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        
        # After initialization, call __read_data__() to manage the data file.
        self.__read_data__()


    def __read_data__(self):
        self.scaler = StandardScaler()

        current_path = os.path.abspath(".")
        print("Current Path:", current_path)

        #get raw data from path
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                        self.data_path))

        # split data set into train, vali, test. border1 is the left border and border2 is the right.
        # Once flag(train, vali, test) is determined, __read_data__ will return certain part of the dataset.
        total_points = len(df_raw)  # total number of data points

        # Definimos los tamaños por proporción
        train_days = int(0.80 * total_points)
        val_days   = int(0.15 * total_points)
        test_days  = total_points - train_days - val_days  # lo que sobra para test

        # Construimos los bordes
        border1s = [
            0,                           # train
            train_days - self.seq_len,  # val (restamos seq_len para asegurar ventana)
            train_days + val_days - self.seq_len  # test
        ]

        border2s = [
            train_days,                  # fin de train
            train_days + val_days,       # fin de val
            total_points                  # fin de test
        ]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        #decide which columns to select
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:] # column name list (remove 'FECHA')
            df_data = df_raw[cols_data]  #remove the first column, which is time stamp info
        elif self.features == 'S':
            df_data = df_raw[[self.target]] # target column

        #scale data by the scaler that fits training data
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            #train_data.values: turn pandas DataFrame into 2D numpy
            self.scaler.fit(train_data.values)  
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values 
        
        # Creo un scaler para el target
        # Este scaler se usará para invertir la escala del target al final
        self.target_scaler = StandardScaler()
        self.target_scaler.fit(train_data[[self.target]].values) 


        #time stamp:df_stamp is a object of <class 'pandas.core.frame.DataFrame'> and
        # has one column called 'date' like 2016-07-01 00:00:00
        df_stamp = df_raw[['FECHA']][border1:border2]
        
        # Since the date format is uncertain across different data file, we need to 
        # standardize it so we call func 'pd.to_datetime'
        df_stamp['FECHA'] = pd.to_datetime(df_stamp.FECHA) 

        if self.timeenc == 0:  #time feature encoding is fixed or learned
            df_stamp['MES'] = df_stamp.FECHA.apply(lambda row: row.month, 1)
            df_stamp['DIA'] = df_stamp.FECHA.apply(lambda row: row.day, 1)
            df_stamp['WEEKDAY'] = df_stamp.FECHA.apply(lambda row: row.weekday(), 1)
            #now df_frame has multiple columns recording the month, day etc. time stamp
            # next we delete the 'date' column and turn 'DataFrame' to a list
            data_stamp = df_stamp.drop(columns=['FECHA'], axis=1).values

        
        # data_x and data_y are same copy of a certain part of data
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2, -1] # last column is the target
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r = s_end  # el target es el siguiente valor

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r:r + self.pred_len]
        if len(seq_y.shape) == 1:
            seq_y = seq_y[:, None]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r:r + self.pred_len]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)