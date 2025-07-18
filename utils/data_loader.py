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
        self.data_list = []  # lista de series
        self.stamp_list = [] # lista de marcas temporales
        self.window_indices = [] # [(serie_idx, start_idx)]
        self.__read_data__()


    def __read_data__(self):
        
        current_path = os.path.abspath(".")
        print("Current Path:", current_path)

        csv_files = [f for f in os.listdir(self.root_path) if f.endswith('.csv')]

        train_data_list_for_scaler = []

        for csv_file in csv_files:
            df_raw = pd.read_csv(os.path.join(self.root_path, csv_file))
            total_points = len(df_raw)

            train_days = int(0.80 * total_points)
            val_days = int(0.15 * total_points)
            test_days = total_points - train_days - val_days

            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                df_data = df_raw[[self.target]]

            # Extraemos solo el train de este CSV
            train_data = df_data.iloc[0:train_days]
            train_data_list_for_scaler.append(train_data.values)

        # Concatenar todo el train para hacer fit del scaler global
        all_train_data = np.concatenate(train_data_list_for_scaler, axis=0)

        # Crear y ajustar escalador global
        self.scaler = StandardScaler()
        self.scaler.fit(all_train_data)

        # Escalador para el target también global
        self.target_scaler = StandardScaler()
        # Extraer solo la última columna (target) de all_train_data para fit
        self.target_scaler.fit(all_train_data[:, -1].reshape(-1,1))



    #----------------------------------------------------------------------------

        for file_idx, csv_file in enumerate(csv_files):
            df_raw = pd.read_csv(os.path.join(self.root_path, csv_file))
            total_points = len(df_raw)
            train_days = int(0.80 * total_points)
            val_days   = int(0.15 * total_points)
            test_days  = total_points - train_days - val_days

            border1s = [
                0,
                train_days - self.seq_len,
                train_days + val_days - self.seq_len
            ]
            border2s = [
                train_days,
                train_days + val_days,
                total_points
            ]
            
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                df_data = df_raw[[self.target]]

            train_data = df_data[border1s[0]:border2s[0]]
            if self.scale:
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values


            df_stamp = df_raw[['FECHA']][border1:border2]
            df_stamp['FECHA'] = pd.to_datetime(df_stamp.FECHA)
            if self.timeenc == 0:
                df_stamp['MES'] = df_stamp.FECHA.apply(lambda row: row.month, 1)
                df_stamp['DIA'] = df_stamp.FECHA.apply(lambda row: row.day, 1)
                df_stamp['WEEKDAY'] = df_stamp.FECHA.apply(lambda row: row.weekday(), 1)
                data_stamp = df_stamp.drop(columns=['FECHA'], axis=1).values

            data_x = data[border1:border2]
            data_y = data[border1:border2, -1]
            self.data_list.append((data_x, data_y))
            self.stamp_list.append(data_stamp)

            # Guardar los índices de ventanas posibles para esta serie
            num_windows = len(data_x) - self.seq_len - self.pred_len + 1
            for i in range(num_windows):
                self.window_indices.append((file_idx, i))


    def __getitem__(self, index):
        serie_idx, start_idx = self.window_indices[index]
        data_x, data_y = self.data_list[serie_idx]
        data_stamp = self.stamp_list[serie_idx]

        s_begin = start_idx
        s_end = s_begin + self.seq_len
        r = s_end

        seq_x = data_x[s_begin:s_end]
        seq_y = data_y[r:r + self.pred_len]
        if len(seq_y.shape) == 1:
            seq_y = seq_y[:, None]

        seq_x_mark = data_stamp[s_begin:s_end]
        seq_y_mark = data_stamp[r:r + self.pred_len]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.window_indices)
    
    def inverse_transform(self, data):
        return self.target_scaler.inverse_transform(data)
