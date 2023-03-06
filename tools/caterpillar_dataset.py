import os
import pandas as pd
import json
import numpy as np

import torch
from torch_geometric.data import Dataset, Data
import torch_geometric.transforms as T
import random


class NormalCaterpillarDataset(Dataset):  # Dataset enforces a very specific file structure. Under 'root' there is 'raw' and 'processed'
    def __init__(self, root, stellar_category, feature_columns, position_columns, use_dataset_ids=None, data_filter=None, repeat=10, label_column=None, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_ids = [1104787, 1130025, 1195075, 1195448, 1232164, 1268839, 1292085,\
                    1354437, 1387186, 1422331, 1422429, 1599988, 1631506, 1631582, 1725139,\
                    1725272, 196589, 264569, 388476, 447649, 5320, 581141, 581180, 65777, 795802,\
                    796175, 94638, 95289, 1079897, 1232423, 1599902, 196078]
        if use_dataset_ids is not None:
            self.dataset_ids = use_dataset_ids
        random.shuffle(self.dataset_ids)
        self.stellar_category = stellar_category
        self.feature_columns = feature_columns
        if position_columns is None:
            self.position_columns = None
        else:
            self.position_columns = sorted(position_columns)
        self.data_filter = data_filter
        self.label_column = label_column
        self.repeat = repeat
        assert pre_filter is None
        super().__init__(root, transform, pre_transform, pre_filter)


    @property
    def raw_file_names(self):
        return [f'labeled_{file_id}_{self.stellar_category}.h5' for file_id in self.dataset_ids]

    @property
    def processed_file_names(self):
        return [f'processed_{i}_{self.stellar_category}.pt' for i in range(len(self.dataset_ids))]

    def process(self):
        for idx, raw_path in enumerate(self.raw_paths):
            df = pd.read_hdf(os.path.join(raw_path), key='star')
            if self.data_filter is not None:
                df = self.data_filter(df)
            positions = torch.tensor(df[self.position_columns].to_numpy()).float()
            with open(raw_path[:-3]+'_norm.json', 'r') as f:
                df_norm = json.load(f)
            features_subs = torch.tensor([df_norm['mean'][feature] for feature in self.feature_columns])
            feature_divs = torch.tensor([df_norm['std'][feature] for feature in self.feature_columns])
            features = torch.tensor(df[self.feature_columns].to_numpy()).float()
            # self.features -= self.features_subs[None,...]     # we don't center data
            features /= feature_divs[None,...]

            if self.label_column is None:
                labels = None
            else:
                labels = torch.tensor(df[self.label_column].to_numpy()).long()
                labels -= torch.min(labels) # first one is always 0

            data = Data(x=features, edge_index=torch.tensor([[],[]]).long(), y=labels, pos=positions)

            if self.pre_transform is not None:     # we cannot build knn here because the whole graph is too large
                data = self.pre_transform(data)


            torch.save(data, os.path.join(self.processed_dir, f'processed_{idx}_{self.stellar_category}.pt'))

    def filter_clusters(self, data, filter_size):  # we will do the zeroing later
        occurrences = torch.bincount(data['y'])
        clusters = torch.nonzero((occurrences <= filter_size) & (occurrences > 0))[:, 0]
        mask = torch.isin(data['y'], clusters)
        data['y'][mask] = -1
        return data

    def sample_space(self, data, radius=0.01, radius_sun=0.0082, zsun_range=0.016/1000, filter_size=10):
        phi = np.random.uniform(0, np.pi*2)
        xsun = np.cos(phi)*radius_sun
        ysun = np.sin(phi)*radius_sun
        zsun = np.random.normal(-zsun_range, zsun_range)
        mask = torch.sum((data['pos'] - torch.tensor([xsun, ysun, zsun])[None, ...])**2, dim=-1) < radius**2
        small_data = data.subgraph(mask)
        if filter_size is not None:
            small_data = self.filter_clusters(small_data, filter_size)
        return small_data

    def add_connectivity(self, data):
        row, col = data['edge_index']
        data['edge_type'] = (data['y'][row] == data['y'][col]) & (data['y'][row] > -1)
        return data

    def len(self):
        return len(self.processed_file_names)*self.repeat

    def get(self, idx):
        if idx % self.repeat == 0:
            self.current_data = torch.load(os.path.join(self.processed_dir, f'processed_{idx//self.repeat}_{self.stellar_category}.pt'))
        data = self.sample_space(self.current_data, radius=0.01, radius_sun=0.0082, zsun_range=0.016/1000, filter_size=10)
        if self.transform is not None:
            data = self.transform(data)
        data = self.add_connectivity(data)
        return data


