import sys
sys.path.append('..')
from sklearn.cluster import DBSCAN, KMeans, AffinityPropagation, SpectralClustering
import torch_geometric.transforms as T
from torch_geometric.data import Dataset, Data
import torch
import numpy as np
from tools.cluster_functions import *
from scipy.sparse import csr_array, csr_matrix





features = torch.tensor(np.random.uniform(size=(70000,13)))
transform = T.KNNGraph(k=300, force_undirected=True)
data = Data(x=features, edge_index=torch.tensor([[],[]]).long(), pos=features)  # pos decides what KNN uses
data = transform(data=data)
edge_pred = torch.exp(-torch.sum((data.x[data.edge_index[0]] - data.x[data.edge_index[1]])**2, dim=-1))
print(edge_pred.shape)
adj = csr_matrix((edge_pred.numpy(), (data.edge_index[0].numpy(), data.edge_index[1].numpy())), shape=(len(data.x), len(data.x)))
#adj = torch.sparse_coo_tensor(data.edge_index.cpu(), edge_pred.cpu(), (len(data.x), len(data.x)))
C_Spectral(adj, n_components=10, eigen_solver='amg')
