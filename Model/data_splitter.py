import torch
import torch.nn.functional as F
import numpy as np
import datetime
import sys
import random
from torch.utils.data import Dataset

class scaler():

    def fit(self,x):
        out = x.permute(2,0,1).contiguous()
        self.mean = torch.mean(out,axis=0)
        self.std = torch.std(out,axis=0)
    
    def transform(self,x):
        out = x.permute(2,0,1).contiguous()
        out = out-self.mean
        out = out/self.std
        return out.permute(1,2,0).contiguous()
    
    def inverse_transform(self,x):
        out = x.permute(2,0,1).contiguous()
        out = out * self.std
        out = out + self.mean
        return out.permute(1,2,0).contiguous()

class ts_dataset(Dataset):
    def __init__(self,mask_index,data_path,datasettype,in_len,out_len,test_len):
        ts = np.load(data_path,allow_pickle=True)
        ts = torch.tensor(ts).float()
        Time_Length = ts.shape[2]
        test_size = test_len - out_len + 1
        train_size = Time_Length-test_len-out_len+1
        test_index = (Time_Length-test_len-in_len,Time_Length-out_len-in_len)
        train_index = (0,Time_Length-test_len-out_len-in_len)
        self.in_len = in_len
        self.out_len = out_len
        self.test_len = test_len
        
        
        train_data = ts[:,:,:Time_Length-test_len]
        test_data = ts[:,:,Time_Length-test_len-in_len:]
        self.scaler = scaler()
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        x=train_data
        x_shape = x.shape
        x=x.reshape(-1,x_shape[-1])
        x[mask_index==1,...]=0
        x = x.reshape(*x_shape)
        x=test_data
        x_shape = x.shape
        x=x.reshape(-1,x_shape[-1])
        x[mask_index==1,...]=0
        x = x.reshape(*x_shape)
        self.time_embedding = torch.arange(Time_Length)%52
        if datasettype == 'train':
            self.data = train_data.permute(2,0,1)
            self.len = Time_Length-test_len-out_len-in_len+1
            self.index = train_index[0]
        else:
            self.data = test_data.permute(2,0,1)
            self.len = test_len-out_len+1
            self.index = test_index[0]
    def __len__(self):
        return self.len
    def __getitem__(self,idx):
        return (self.data[idx:idx+self.in_len,:,:],self.time_embedding[self.index+idx:self.index+idx+self.in_len],self.data[idx+self.in_len:idx+self.in_len+self.out_len,:,:])


# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# train_set = ts_dataset(datas,industry,'train',52,4,10)
# test_set = ts_dataset(datas,industry,'test',52,4,10)