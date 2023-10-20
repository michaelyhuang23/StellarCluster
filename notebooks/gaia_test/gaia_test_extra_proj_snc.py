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
from tools.gnn_models import GCNEdgeBased, FeedForwardProjector
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


# %%
def train_projector(edge_index, edge_pred, n_components=5, EPOCHS=100):
    projector = FeedForwardProjector(len(feature_columns), [64, 64], n_components=n_components).to(device)
    optimizer = Adam(projector.parameters(), lr=0.001)
    projector.train()
    for epoch in range(EPOCHS):
        for graph in gaia_loader:
            optimizer.zero_grad()
            graph = graph.to(device)
            FX = projector(graph.x)
            loss = projector.edge_pred_loss(edge_index, edge_pred, FX)
            acc = projector.edge_pred_acc(edge_index, edge_pred, FX)
            writer.add_scalar('Proj/Loss', loss.item(), epoch*len(gaia_loader)+i)
            writer.add_scalar('Proj/EdgeAcc', acc.item(), epoch*len(gaia_loader)+i)
            loss.backward()
            optimizer.step()
    return projector


# %%
import os

projector = train_projector(t_edge_index, t_edge_pred, n_components=5, EPOCHS=100)

data_root = '../../data/gaia/raw/'
results_root = '../../results/cluster_files/'

df = pd.read_hdf(os.path.join(data_root, 'common_kinematics_10kpc.h5'), key='star')
np.random.seed(0)
keep_idx = np.random.choice(len(df), 3000, replace=False)
df = df.iloc[keep_idx]
features = torch.tensor(df[feature_columns].to_numpy()).float()

FX = projector(features.to(device)).cpu().detach().numpy()
cluster_labels = np.argmax(FX, axis=1)

labels = pd.DataFrame(cluster_labels, columns=['cluster_id'])
labels.to_csv(os.path.join(results_root, 'gaia_projection_snc_mom_3000.csv'), index=False)

print('done')