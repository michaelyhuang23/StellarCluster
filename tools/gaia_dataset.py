import os
import pandas as pd
import json
import numpy as np

import torch
from torch_geometric.data import Dataset, Data
import torch_geometric.transforms as T
import random


class SampleGaiaDataset(Dataset):  # Dataset enforces a very specific file structure. Under 'root' there is 'raw' and 'processed'
    def __init__(self, root, feature_columns, sample_size=10000, num_samples=10, transform=None, pre_transform=None):
        self.feature_columns = feature_columns
        self.sample_size = sample_size
        self.num_samples = num_samples
        self.feature_divs = None
        self.features_subs = None
        super().__init__(root, transform, pre_transform, False) # no pre-filter

    @property
    def raw_file_names(self):
        return ['common_kinematics_10kpc.h5']

    @property
    def processed_file_names(self):
        return [f'processed_{idx}_gaia_10kpc.pt' for idx in range(self.num_samples)]

    def get_normalizer(self):
        raw_path = self.raw_paths[0]
        with open(raw_path[:-3]+'_norm.json', 'r') as f:
            df_norm = json.load(f)
        self.features_subs = torch.tensor([df_norm['mean'][feature] for feature in self.feature_columns])
        self.feature_divs = torch.tensor([df_norm['std'][feature] for feature in self.feature_columns])
        self.df_norm = df_norm

    def de_normalize(self, X):
        if self.feature_divs is not None:
            return X * self.feature_divs
        else:
            self.get_normalizer()
            return X * self.feature_divs

    def process(self):
        raw_path = self.raw_paths[0]
        df = pd.read_hdf(os.path.join(raw_path), key='star')
        self.get_normalizer()
        if 'Etot' in self.feature_columns:
            df['Etot'] -= self.df_norm['mean']['Etot']
        features = torch.tensor(df[self.feature_columns].to_numpy()).float()
        # self.features -= self.features_subs[None,...]     # we don't center data
        features /= self.feature_divs[None,...]
        stellar_ids = torch.tensor(df['source_id'].to_numpy()).long()

        if len(features) <= self.sample_size:
            data = Data(x=features, edge_index=torch.tensor([[],[]]).long(), y=None, pos=features, ids=stellar_ids, total_size=len(features))  # pos decides what KNN uses
            if self.pre_transform is not None:  # we should build knn here instead of at transform
                data = self.pre_transform(data)
            torch.save(data, os.path.join(self.processed_dir, f'processed_0_gaia_10kpc.pt'))
        else: 
            for idx in range(self.num_samples):
                sampled_idx = np.random.choice(len(features), self.sample_size, replace=False)
                sampled_stellar_ids = stellar_ids[sampled_idx]
                data = Data(x=features[sampled_idx], edge_index=torch.tensor([[],[]]).long(), y=None, pos=features[sampled_idx], ids=sampled_stellar_ids, sampled_idx=sampled_idx, total_size=len(features))  # pos decides what KNN uses
                if self.pre_transform is not None:  # we should build knn here instead of at transform
                    data = self.pre_transform(data)
                torch.save(data, os.path.join(self.processed_dir, f'processed_{idx}_gaia_10kpc.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'processed_{idx}_gaia_10kpc.pt'))
        return data


class GaiaDataset(Dataset):  # Dataset enforces a very specific file structure. Under 'root' there is 'raw' and 'processed'
    def __init__(self, root, feature_columns, transform=None, pre_transform=None):
        self.feature_columns = feature_columns
        self.feature_divs = None
        self.features_subs = None
        super().__init__(root, transform, pre_transform, False) # no pre-filter

    @property
    def raw_file_names(self):
        return ['common_kinematics_10kpc.h5']

    @property
    def processed_file_names(self):
        return ['processed_0_gaia_10kpc.pt']

    def get_normalizer(self):
        raw_path = self.raw_paths[0]
        with open(raw_path[:-3]+'_norm.json', 'r') as f:
            df_norm = json.load(f)
        self.features_subs = torch.tensor([df_norm['mean'][feature] for feature in self.feature_columns])
        self.feature_divs = torch.tensor([df_norm['std'][feature] for feature in self.feature_columns])

    def de_normalize(self, X):
        if self.feature_divs is not None:
            return X * self.feature_divs
        else:
            self.get_normalizer()
            return X * self.feature_divs

    def process(self):
        raw_path = self.raw_paths[0]
        df = pd.read_hdf(os.path.join(raw_path), key='star')
        self.get_normalizer()
        features = torch.tensor(df[self.feature_columns].to_numpy()).float()
        # self.features -= self.features_subs[None,...]     # we don't center data
        features /= self.feature_divs[None,...]

        data = Data(x=features, edge_index=torch.tensor([[],[]]).long(), y=None, pos=features)  # pos decides what KNN uses
        if self.pre_transform is not None:  # we should build knn here instead of at transform
            data = self.pre_transform(data)
        torch.save(data, os.path.join(self.processed_dir, 'processed_0_gaia_10kpc.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'processed_{idx}_gaia_10kpc.pt'))
        return data
