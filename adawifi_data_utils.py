#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import io
import os
import sys
import h5py
import copy
import random
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.io as scio
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from datetime import datetime, timedelta
import time
import seaborn as sns
import pywt
import scipy.signal

GESTURES = ['Swipe', 'Push-Pull', 'Clap', 'Slide', 'Draw-Z', 'Draw-O']
CSI_30_INDICES = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 27, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 55]


class DfsData(object):
    def __init__(self, dir):
        self.dir = dir
        self.dfs_dir = self.dir # os.path.join(widar_dir, 'DFS')
        self.data_all = {}
        self.load_dfs_dir()
        self.df = pd.DataFrame(self.data_all.values())

    def load_dfs_dir(self):
        dfs_data_all = {}
        for data_root, data_dirs, data_files in os.walk(self.dfs_dir):
            for data_file_name in data_files:
                file_path = os.path.join(data_root, data_file_name)
                if not file_path.endswith('.mat'):
                    continue
                user = os.path.basename(os.path.dirname(data_root))
                seps = data_file_name.split('.')[0].split('_')
                location = data_root.split('/')[-3]
                gesture_name = seps[0]
                timestamp = seps[1]
                data_id = f"{gesture_name}-{user}-{location}-{timestamp}"
                # print(data_id)
                # input('pause')
                data_dict = {
                    'date': user,
                    'location': location,
                    'gesture': gesture_name,
                    'timestamp': int(timestamp),
                    'data_id': data_id,
                    'dfs_path': file_path
                }
                dfs_data_all[data_id] = data_dict

        self.data_all = dfs_data_all


class CsiData(object):
    def __init__(self, dir):
        self.dir = dir
        self.csi_dir = self.dir # os.path.join(widar_dir, 'DFS')
        self.data_all = {}
        self.load_csi_dir()
        self.df = pd.DataFrame(self.data_all.values())

    def load_csi_dir(self):
        csi_data_all = {}
        for data_root, data_dirs, data_files in os.walk(self.csi_dir):
            for data_file_name in data_files:
                file_path = os.path.join(data_root, data_file_name)
                if not file_path.endswith('.npy'):
                    continue
                date = os.path.basename(os.path.dirname(data_root))
                seps = data_file_name.split('.')[0].split('_')
                gesture_name = seps[0]
                timestamp = seps[1]
                data_id = f"{date}-{timestamp}"
                data_dict = {
                    'date': date,
                    'gesture': gesture_name,
                    'timestamp': int(timestamp),
                    'data_id': data_id,
                    'csi_path': file_path
                }
                csi_data_all[data_id] = data_dict

        self.data_all = csi_data_all


class DfsSampleToTensor(object):
    def __call__(self, sample):
        return {
            'dfs': torch.from_numpy(sample['dfs']).float(),
            'gesture_id': torch.tensor(sample['gesture_id'], dtype=torch.long),
            'data_T': torch.tensor(sample['data_T'], dtype=torch.long),
            'gesture': sample['gesture'],
            'timestamp': sample['timestamp'],
        }


class CsiSampleToTensor(object):
    def __call__(self, sample):
        return {
            'csi': torch.from_numpy(sample['csi']).float(),
            'gesture_id': torch.tensor(sample['gesture_id'], dtype=torch.long),
            'data_T': torch.tensor(sample['data_T'], dtype=torch.long),
            'gesture': sample['gesture'],
            'timestamp': sample['timestamp'],
        }


def load_dfs_mat(file_path):
    dfs_data = None
    try:
        data = sio.loadmat(file_path)
        dfs_data = data['doppler_spectrum']
    except Exception as e:
        try:
            f = h5py.File(file_path, 'r')
            data = {}
            for k, v in f.items():
                data[k] = np.array(v)
            dfs_data = data['doppler_spectrum'].swapaxes(0, 2)
            # print('loaded with h5py')
            # print(data)
        except Exception as e:
            # print('failed to load with h5py')
            pass
    return dfs_data


def load_dfs_with_cache(data_id, cache_dir, dfs_path, do_slice=True):
    cache_dfs_path = os.path.join(cache_dir, 'dfs', f'{data_id}_dfs.npy')
    if not (os.path.exists(cache_dfs_path)):
        dfs = load_dfs_mat(dfs_path)
        if dfs is None:
            print('DFS is none!')
            return dfs
        def circshift1D(lst, k):
            return np.concatenate([lst[-k:], lst[: -k]])
        for i in range(dfs.shape[0]):
            dfs[i] = circshift1D(np.abs(dfs[i]), int(np.size(dfs[i], 0) / 2))
        if do_slice:
            dfs = dfs[:, :, ::50]
        np.save(cache_dfs_path, dfs)
    dfs = np.load(cache_dfs_path)
    return dfs


class DfsDataset(Dataset):
    def __init__(self, df, transform=None, target_gestures=None, receivers=None, cache_dir='temp', silient=False):
        self.df = df
        self.all_gestures = sorted(set(df['gesture']))
        self.target_gestures = target_gestures if target_gestures else self.all_gestures
        self.receivers = receivers
        self.silient = silient
        self.cache_dir = cache_dir
        os.makedirs(os.path.join(self.cache_dir, 'dfs'), exist_ok=True)
        self.all_samples = self.load_data(df)
        self.transform = transform if transform else DfsSampleToTensor()

    def load_data(self, df):
        df = df.sort_values(['gesture'], ascending=True)
        all_samples = []
        for i, r in tqdm(df.iterrows(), desc='load samples', total=len(df), disable=self.silient):
            if r.gesture not in self.target_gestures:
                continue
            gesture_id = self.target_gestures.index(r.gesture)
            # load_mat = scio.loadmat(r.dfs_path)
            # dfs = load_mat['doppler_spectrum']
            # def circshift1D(lst, k):
            #     return np.concatenate([lst[-k:], lst[: -k]])
            # for i in range(dfs.shape[0]):
            #     dfs[i] = circshift1D(np.abs(dfs[i]), int(np.size(dfs[i], 0) / 2))
            # dfs = dfs[:, :, ::50]
            dfs = load_dfs_with_cache(r.data_id, cache_dir=self.cache_dir, dfs_path=r.dfs_path)
            if dfs is None:
                continue
            sample = {
                'timestamp': r.timestamp,
                'gesture': r.gesture,
                'gesture_id': gesture_id,
                'data_T': dfs.shape[-1],
                'dfs': dfs
            }
            all_samples.append(sample)

        return all_samples

    def __getitem__(self, index):
        sample = self.all_samples[index]
        if self.receivers is not None:
            sample['dfs'] = sample['dfs'][self.receivers]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.all_samples)


def collate_fn_dfs(samples):
    # cut_T = max([sample['dfs'].shape[-1] for sample in samples])
    cut_T = 60
    for sample in samples:
        data = sample['dfs']
        # data = data[0:2,:,:]
        if cut_T <= data.shape[-1]:
            data = data[:, :, 0:cut_T]
        data = F.pad(data, (0, cut_T - data.shape[-1]))
        if_nan = np.isnan(data)
        if True in if_nan:
            data[np.isnan(data)] = 0
        sample['dfs'] = data
    return default_collate(samples)


def collate_fn_csi(samples):
    cut_T = max([sample['csi'].shape[-1] for sample in samples])
    for sample in samples:
        data = sample['csi']
        if cut_T <= data.shape[-1]:
            data = data[:, :, 0:cut_T]
        data = F.pad(data, (0, cut_T - data.shape[-1]))
        if_nan = np.isnan(data)
        if True in if_nan:
            data[np.isnan(data)] = 0
        sample['csi'] = data
    return default_collate(samples)


def timestamp_datetime(timestamp):
    time_local = time.localtime(int(timestamp))
    dt = time.strftime("%Y%m%d_%H%M%S", time_local)
    dt = datetime.strptime(dt, '%Y%m%d_%H%M%S')
    return dt

            
class MyDfsDataset(Dataset):
    def __init__(self, df, target_gestures, transform=None, cache_dir='my_dfs_cache', silient=False):
        self.df = df
        self.target_gestures = target_gestures
        self.cache_dir = cache_dir
        os.makedirs(os.path.join(cache_dir, 'dfs'), exist_ok=True)
        self.silient = silient
        self.all_samples = self.load_data(df)
        self.transform = transform if transform else MyDfsDataset.default_transform

    def load_data(self, df):
        df = df.sort_values(['gesture'], ascending=True)
        all_samples = []
        for i, r in tqdm(df.iterrows(), desc='load samples', total=len(df), disable=self.silient):
            try:
                dfs = load_dfs_with_cache(r.data_id, cache_dir=self.cache_dir, dfs_path=r.dfs_path, do_slice=False)
                if dfs is None:
                    continue
                if_nan = np.isnan(dfs)
                if True in if_nan:
                    dfs[np.isnan(dfs)] = 0

                sample = r.to_dict()
                sample['gesture_id'] = self.target_gestures.index(r.gesture)
                sample['dfs'] = dfs
                all_samples.append(sample)
            except:
                pass
        return all_samples
    
    @staticmethod
    def default_transform(sample):
        dfs = sample['dfs']
        dfs = dfs[:, :, ::100]
        return {
            'gesture_id': torch.tensor(sample['gesture_id'], dtype=torch.long),
            'dfs': torch.from_numpy(dfs).float(),
            'data_T': torch.tensor(dfs.shape[-1], dtype=torch.long)
        }
    
    @staticmethod
    def augment_transform(sample):
        dfs = sample['dfs']
        rand_timestamp = np.random.randint(1, 300)
        rand_interval = np.random.randint(90, 110)
        dfs = dfs[:, :, :-rand_timestamp][:, :, ::rand_interval]
        return {
            'gesture_id': torch.tensor(sample['gesture_id'], dtype=torch.long),
            'dfs': torch.from_numpy(dfs).float(),
            'data_T': torch.tensor(dfs.shape[-1], dtype=torch.long)
        }
    
    @staticmethod
    def default_collate_fn(samples):
        cut_T = max([sample['dfs'].shape[-1] for sample in samples])
        n_clients = max([sample['dfs'].shape[0] for sample in samples])
        for sample in samples:
            data = sample['dfs']
            if cut_T <= data.shape[-1]:
                data = data[:, :, 0:cut_T]
            data = F.pad(data, (0, cut_T - data.shape[-1], 0, 0, 0, n_clients - data.shape[0]))
            sample['dfs'] = data
            if 'pos' in sample:
                pos = sample['pos']
                sample['pos'] = F.pad(pos, (0, n_clients - pos.size(0)))
        return default_collate(samples)

    def __getitem__(self, idx):
        sample = self.all_samples[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.all_samples)
            


if __name__ == '__main__':
    target_gestures = ['Swipe', 'Push-Pull', 'Clap', 'Slide', 'Draw-Z', 'Draw-O']
   

