import sys
sys.path.append('..')

import torch
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, AffinityPropagation, SpectralClustering
from sklearn.mixture import GaussianMixture
from hdbscan import HDBSCAN
from torch.optim import SGD, Adam


from tools.gnn_models import *

def C_HDBSCAN(data, min_cluster_size=5, min_samples=None, cluster_selection_method='eom'):
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_method=cluster_selection_method)
    clusterer.fit(data)
    return clusterer.labels_

def C_Spectral(adj, n_components=30):
    clusterer = SpectralClustering(n_components, affinity='precomputed', assign_labels='discretize')
    clusterer.fit(adj)
    return clusterer.labels_

def C_GaussianMixture(data, n_components):
    clusterer = GaussianMixture(n_components=n_components, covariance_type='full')
    return clusterer.fit_predict(data)

def C_Edge2Cluster(data, edge_pred, n_components, cluster_lr=0.003, cluster_regularizer=0.00001, device='cpu'):
    model = GCNEdge2Cluster(data.x.shape[-1], num_cluster=n_components, regularizer=cluster_regularizer).to(device)
    optim = Adam(model.parameters(), lr=cluster_lr, weight_decay=1e-5)
    model.train()
    for i in range(4000):
        optim.zero_grad()
        FX, loss = model(data, edge_pred)
        loss.backward()
        optim.step()
        if (i+1)%100 == 0:
            print('cluster loss:',loss.item())
    FX, loss = model(data)
    return FX.detach().cpu()

def T_Edge2Cluster(data, edge_pred, n_components, gap=100, cluster_lr=0.003, cluster_regularizer=0.00001, device='cpu'):
    model = GCNEdge2Cluster(data.x.shape[-1], num_cluster=n_components, regularizer=cluster_regularizer).to(device)
    optim = Adam(model.parameters(), lr=cluster_lr, weight_decay=1e-5)
    model.train()
    for i in range(4000):
        optim.zero_grad()
        FX, loss = model(data, edge_pred)
        loss.backward()
        optim.step()
        if (i+1)%gap == 0:
            print('cluster loss:',loss.item())
            yield FX.detach().cpu()
