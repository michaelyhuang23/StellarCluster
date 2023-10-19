# %%
import sys
sys.path.append('../..')

import json
import pandas as pd

import torch
from torch.optim import SGD, Adam
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch.utils.tensorboard import SummaryWriter
from tools.gaia_dataset import SampleGaiaDataset
from tools.gnn_models import GCNEdgeBased
from tools.evaluation_metric import *
from tools.cluster_functions import *

# %%
writer = SummaryWriter()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

feature_columns = ['Etot', 'JR', 'Jz', 'Jphi'] #, 'Vtot','W', 'vr', 'vphi'] #, 'RGC', 'Vtot', 'U', 'V', 'W', 'vr', 'vphi', 'PhiGC', 'ZGC']
position_columns = ['XGC', 'YGC', 'ZGC']

# %%
data_transforms = T.Compose(transforms=[T.KNNGraph(k=300, force_undirected=True, loop=False), T.GDC(normalization_out='sym', self_loop_weight=None, sparsification_kwargs={'avg_degree':300, 'method':'threshold'})]) 
gaia_dataset = SampleGaiaDataset('../../data/gaia', feature_columns, sample_size=2000, num_samples=40, pre_transform=data_transforms) 
# num_samples need to be at least as big as (total_size / sample_size)**2. Which is around 625 in this case.
gaia_loader = DataLoader(gaia_dataset, batch_size=1, shuffle=True)

# %%
model = GANOrigEdgeBased(len(feature_columns), regularizer=0).to(device)
model.load_state_dict(torch.load('../../train_script/weights/GANOrigEdgeBased_model300new_gaia_mom/299.pth', map_location=device)['model_state_dict'])

# %%
from scipy.sparse import csr_matrix
def evaluate(graph, model):
    graph = graph.to(device)
    with torch.no_grad():
        model.eval()
        edge_pred = model(graph)
    edge_index = graph.sampled_idx[graph.edge_index].cpu()
    adj = torch.sparse_coo_tensor(edge_index, edge_pred.cpu(), (graph.total_size, graph.total_size)) 
    return adj, graph.sampled_idx.cpu().numpy(), graph.ids.cpu().numpy()

# %%
print('computing adjacency matrix')
t_adj = None
stellar_ids = None
for i, graph in enumerate(gaia_loader):
    if stellar_ids is None: 
        stellar_ids = np.zeros((graph.total_size), dtype=np.int64)
    adj, sampled_idx, ids = evaluate(graph, model)
    stellar_ids[sampled_idx] = ids
    if t_adj is None:
        t_adj = adj
    else:
        t_adj = t_adj*i/(i+1) + adj/(i+1) 
t_adj = t_adj.coalesce()

t_edge_index = t_adj.indices()
t_edge_pred = t_adj.values()

# sparsify t_adj
keep_edges = 10000000
if len(t_adj.values()) > keep_edges:
    keep_edge_idx = np.random.choice(len(t_adj.values()), keep_edges, replace=False)
    t_edge_index = t_adj.indices()[:, keep_edge_idx]
    t_edge_pred = t_adj.values()[keep_edge_idx]

print(f'number of edges: {len(t_edge_index[0])}')

adj = csr_matrix((t_edge_pred, t_edge_index), shape=t_adj.shape)

# %%

print('performing clustering')
print(f'adj shape: {adj.shape}')
# perform clustering
n_components = 5
FX = C_Spectral(adj, n_components=n_components)


labels = pd.DataFrame(FX, columns=['cluster_id'])
labels['source_id'] = stellar_ids
labels.to_csv('../../results/cluster_files/gaia_stitch_snc_mom_random_sparsification.csv', index=False)

print('done')

#
## %%
#import seaborn as sns
#import matplotlib.pyplot as plt
#sns.set_palette(sns.color_palette("colorblind"))
#
## %%
#local_graph = graph.to('cpu')
#X = gaia_dataset.de_normalize(local_graph.x)
#
## %%
#from collections import Counter
#counts = Counter(clusters).most_common()
#top2 = [key for (key, c) in counts[:]]
#print(top2)
#
## %%
#X = X[np.isin(clusters, top2)]
#clusters = [cluster for cluster in clusters if cluster in top2]
#
## %%
#df_x = pd.DataFrame(X, columns = feature_columns)
#
#
## %%
#fig, axs = plt.subplots(ncols=3,nrows=2,figsize=(15, 10))
#axs = axs.flatten()
#sns.scatterplot(data=df_x, x='Jphi', y='Etot', hue=clusters, style=clusters, s=10, ax=axs[0])
#sns.scatterplot(data=df_x, x='vr', y='vphi', hue=clusters, style=clusters, s=10, ax=axs[1])
#sns.scatterplot(data=df_x, x='vr', y='W', hue=clusters, style=clusters, s=10, ax=axs[2])
#J = np.sqrt(X[:,3]**2 + X[:,2]**2 + X[:,1]**2)
#sns.scatterplot(x=X[:,3]/J, y=(X[:,2]-X[:,1])/J, hue=clusters, style=clusters, s=10, ax=axs[3])
#sns.scatterplot(data=df_x, x='JR', y='Jphi', hue=clusters, style=clusters, s=10, ax=axs[4])
#sns.scatterplot(data=df_x, x='JR', y='Jz', hue=clusters, style=clusters, s=10, ax=axs[5])
#fig.savefig('../../gaia_full_snc.png')
#
## %%
#fig, axs = plt.subplots(ncols=3,nrows=2,figsize=(15, 10))
#axs = axs.flatten()
#sns.scatterplot(data=df_x, x='Jphi', y='Etot', hue=clusters, style=clusters, s=10, ax=axs[0])
#J = np.sqrt(X[:,3]**2 + X[:,2]**2 + X[:,1]**2)
#sns.scatterplot(x=X[:,3]/J, y=(X[:,2]-X[:,1])/J, hue=clusters, style=clusters, s=10, ax=axs[3])
#sns.scatterplot(data=df_x, x='JR', y='Jphi', hue=clusters, style=clusters, s=10, ax=axs[4])
#sns.scatterplot(data=df_x, x='JR', y='Jz', hue=clusters, style=clusters, s=10, ax=axs[5])
#fig.savefig('../../gaia_full_snc.png')
#
## %%
#sns.scatterplot(x=X[:,9], y=X[:,10], hue=clusters, style=clusters, s=10)
#
## %%
#J = np.sqrt(X[:,3]**2 + X[:,2]**2 + X[:,1]**2)
#ax = sns.scatterplot(x=X[:,3]/J, y=(X[:,2]-X[:,1])/J, hue=clusters, style=clusters, s=10)
#
## %%
#ax = sns.scatterplot(x=X[:,1], y=X[:,2], hue=clusters, style=clusters, s=10)
#ax.set(xlim=[0,20000])
#
## %%
#ax = sns.scatterplot(x=X[:,4], y=X[:,12], hue=clusters, style=clusters, s=10)
#
## %%
#POS = [X[:,4]*torch.cos(X[:,11]),X[:,4]*torch.sin(X[:,11]),X[:,12]]
#ax = sns.scatterplot(x=POS[0], y=POS[1], hue=clusters, style=clusters, s=10)
#
## %%
#ax = sns.scatterplot(x=X[:,11], y=X[:,4], hue=clusters, style=clusters, s=10)
#
## %%
#
#
## %%
#
#
## %%




# %%
