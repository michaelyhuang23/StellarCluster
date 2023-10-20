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
    return edge_pred 

# %%
print('computing new graph')
edge_preds = []
for i, graph in enumerate(gaia_dataset):
    edge_pred = evaluate(graph, model)
    edge_preds.append(edge_pred)

# %%
def train_projector(dataset, edge_preds, n_components=5, EPOCHS=100):
    projector = FeedForwardProjector(len(feature_columns), [64, 64], n_components=n_components).to(device)
    optimizer = Adam(projector.parameters(), lr=0.001)
    projector.train()
    for epoch in range(EPOCHS):
        mean_loss = 0
        for i, graph in enumerate(dataset):
            optimizer.zero_grad()
            graph = graph.to(device)
            FX = projector(graph.x)
            edge_pred = edge_preds[i].to(device)
            loss = projector.edge_pred_loss(graph.edge_index, edge_pred, FX)
            acc = projector.edge_pred_acc(graph.edge_index, edge_pred, FX)
            writer.add_scalar('Proj/Loss', loss.item(), epoch*len(gaia_dataset)+i)
            writer.add_scalar('Proj/EdgeAcc', acc.item(), epoch*len(gaia_dataset)+i)
            loss.backward()
            optimizer.step()
            mean_loss += loss.item()
        mean_loss /= len(dataset)
        print(f'Epoch {epoch}: loss {mean_loss}')
    return projector


# %%
import os

projector = train_projector(gaia_dataset, edge_preds, n_components=5, EPOCHS=100)

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