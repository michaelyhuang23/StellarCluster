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
        super().__init__(root, transform, pre_transform, False) # no pre-filter

    @property
    def raw_file_names(self):
        return ['common_kinematics_10kpc.h5']

    @property
    def processed_file_names(self):
        return [f'processed_{idx}_gaia_10kpc.pt' for idx in range(self.num_samples)]

    def process(self):
        raw_path = self.raw_paths[0]
        df = pd.read_hdf(os.path.join(raw_path), key='star')
        with open(raw_path[:-3]+'_norm.json', 'r') as f:
            df_norm = json.load(f)
        features_subs = torch.tensor([df_norm['mean'][feature] for feature in self.feature_columns])
        feature_divs = torch.tensor([df_norm['std'][feature] for feature in self.feature_columns])
        features = torch.tensor(df[self.feature_columns].to_numpy()).float()
        # self.features -= self.features_subs[None,...]     # we don't center data
        features /= feature_divs[None,...]

        for idx in range(self.num_samples):
            sampled_ids = np.random.choice(len(features), self.sample_size)
            data = Data(x=features[sampled_ids], edge_index=torch.tensor([[],[]]).long(), y=None, pos=features[sampled_ids], sample_indices=sampled_ids)  # pos decides what KNN uses
            if self.pre_transform is not None:  # we should build knn here instead of at transform
                data = self.pre_transform(data)
            torch.save(data, os.path.join(self.processed_dir, f'processed_{idx}_gaia_10kpc.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'processed_{idx}_gaia_10kpc.pt'))
        return data


