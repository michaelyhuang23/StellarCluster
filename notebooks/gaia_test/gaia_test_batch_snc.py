#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../..')

import json
import pandas as pd

import torch
from torch.optim import SGD, Adam
from torch_geometric.loader import DataLoader
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
import torch_geometric.transforms as T
from torch.utils.tensorboard import SummaryWriter
from tools.gaia_dataset import SampleGaiaDataset, GaiaDataset
from tools.gnn_models import GCNEdgeBased
from tools.evaluation_metric import *
from tools.cluster_functions import *


# In[2]:


writer = SummaryWriter()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

feature_columns = ['Etot', 'JR', 'Jz', 'Jphi']
position_columns = ['XGC', 'YGC', 'ZGC']


# In[3]:


data_transforms = T.Compose(transforms=[T.KNNGraph(k=100, force_undirected=True)]) 
gaia_dataset = GaiaDataset('../../data/gaia', feature_columns, pre_transform=data_transforms)


# In[4]:


total_graph = gaia_dataset[0]


# In[5]:


def add_indices(data_batch): # complicated tensor ops to map edges in edge_label_index back to indices of edges in edge_index
    n = len(data_batch.x)
    edge_label_index = data_batch.edge_label_index
    edge_index = data_batch.edge_index

    edge_label_index_order = edge_label_index[0] * n + edge_label_index[1]
    edge_index_order = edge_index[0] * n + edge_index[1]

    edge_index_order = torch.concat([edge_index_order, edge_label_index_order])
    edge_index_order, indices = torch.unique(edge_index_order, sorted=True, return_inverse=True)

    edge_index = torch.concat([edge_index, edge_label_index], dim=-1)
    new_edge_index = torch.zeros((2, len(edge_index_order))).long()
    new_edge_index[:, indices] = edge_index
    data_batch.edge_index = new_edge_index

    if 'edge_type' in data_batch.keys:
        edge_type = torch.concat([data_batch.edge_type, data_batch.edge_label])  # must have valid edge_labels
        new_edge_type = torch.zeros((len(edge_index_order))).bool()
        new_edge_type[indices] = edge_type
        data_batch.edge_type = new_edge_type

    match_id = torch.searchsorted(edge_index_order, edge_label_index_order)
    data_batch.match_id = match_id

    return data_batch


# In[6]:


edge_count = len(total_graph.edge_index)
gaia_loader = LinkNeighborLoader(total_graph, num_neighbors=[100, 100], batch_size=8192, edge_label_index=total_graph.edge_index, transform=add_indices)


# In[7]:


model = GANOrigEdgeBased(len(feature_columns), regularizer=0).to(device)
model.load_state_dict(torch.load('../../train_script/weights/GANOrigEdgeBased_model300new_gaia_mom/299.pth', map_location=torch.device('cpu'))['model_state_dict'])


# In[8]:


from scipy.sparse import csr_matrix
def evaluate(n_components, total_graph, graph_loader, model):
    total_edge_pred = []
    total_edge_index = []
    for i,graph in enumerate(graph_loader):
        graph = graph.to(device)
        print(i,graph)
        with torch.no_grad():
            model.eval()
            edge_pred = model(graph)
        total_edge_pred.append(edge_pred[graph.match_id].cpu())
        total_edge_index.append(graph.edge_label_index.cpu())
    t_edge_pred = torch.concat(total_edge_pred)
    t_edge_index = torch.concat(total_edge_index, dim=-1)
    adj = csr_matrix((t_edge_pred,t_edge_index), shape=(len(total_graph.x), len(total_graph.x)))    
    FX = C_Spectral(adj, n_components=n_components)
    return FX


# In[ ]:


FX = evaluate(5, total_graph, gaia_loader, model)
clusters = [f'cluster {idx}' for idx in FX]


# In[63]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set_palette(sns.color_palette("colorblind"))


# In[64]:


local_graph = graph.to('cpu')
X = gaia_dataset.de_normalize(local_graph.x)


# In[65]:


from collections import Counter
counts = Counter(clusters).most_common()
top2 = [key for (key, c) in counts[:]]
print(top2)


# In[66]:


X = X[np.isin(clusters, top2)]
clusters = [cluster for cluster in clusters if cluster in top2]


# In[67]:


df_x = pd.DataFrame(X, columns = feature_columns)


# In[68]:


sns.scatterplot(x=X[:,3], y=X[:,0], s=10)


# In[19]:


fig, axs = plt.subplots(ncols=3,nrows=2,figsize=(15, 10))
axs = axs.flatten()
sns.scatterplot(data=df_x, x='Jphi', y='Etot', hue=clusters, style=clusters, s=10, ax=axs[0])
sns.scatterplot(data=df_x, x='vr', y='vphi', hue=clusters, style=clusters, s=10, ax=axs[1])
sns.scatterplot(data=df_x, x='vr', y='W', hue=clusters, style=clusters, s=10, ax=axs[2])
J = np.sqrt(X[:,3]**2 + X[:,2]**2 + X[:,1]**2)
sns.scatterplot(x=X[:,3]/J, y=(X[:,2]-X[:,1])/J, hue=clusters, style=clusters, s=10, ax=axs[3])
sns.scatterplot(data=df_x, x='JR', y='Jphi', hue=clusters, style=clusters, s=10, ax=axs[4])
sns.scatterplot(data=df_x, x='JR', y='Jz', hue=clusters, style=clusters, s=10, ax=axs[5])


# In[69]:


fig, axs = plt.subplots(ncols=3,nrows=2,figsize=(15, 10))
axs = axs.flatten()
sns.scatterplot(data=df_x, x='Jphi', y='Etot', hue=clusters, style=clusters, s=10, ax=axs[0])
J = np.sqrt(X[:,3]**2 + X[:,2]**2 + X[:,1]**2)
sns.scatterplot(x=X[:,3]/J, y=(X[:,2]-X[:,1])/J, hue=clusters, style=clusters, s=10, ax=axs[3])
sns.scatterplot(data=df_x, x='JR', y='Jphi', hue=clusters, style=clusters, s=10, ax=axs[4])
sns.scatterplot(data=df_x, x='JR', y='Jz', hue=clusters, style=clusters, s=10, ax=axs[5])


# In[ ]:


sns.scatterplot(x=X[:,9], y=X[:,10], hue=clusters, style=clusters, s=10)


# In[ ]:


J = np.sqrt(X[:,3]**2 + X[:,2]**2 + X[:,1]**2)
ax = sns.scatterplot(x=X[:,3]/J, y=(X[:,2]-X[:,1])/J, hue=clusters, style=clusters, s=10)


# In[ ]:


ax = sns.scatterplot(x=X[:,1], y=X[:,2], hue=clusters, style=clusters, s=10)
ax.set(xlim=[0,20000])


# In[ ]:


ax = sns.scatterplot(x=X[:,4], y=X[:,12], hue=clusters, style=clusters, s=10)


# In[ ]:


POS = [X[:,4]*torch.cos(X[:,11]),X[:,4]*torch.sin(X[:,11]),X[:,12]]
ax = sns.scatterplot(x=POS[0], y=POS[1], hue=clusters, style=clusters, s=10)


# In[ ]:


ax = sns.scatterplot(x=X[:,11], y=X[:,4], hue=clusters, style=clusters, s=10)


# In[ ]:





# In[ ]:





# In[ ]:




