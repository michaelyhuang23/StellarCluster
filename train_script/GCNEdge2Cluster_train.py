import sys
sys.path.append('..')

import torch
from torch.optim import SGD, Adam
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch.utils.tensorboard import SummaryWriter
from tools.caterpillar_dataset import NormalCaterpillarDataset
from tools.gnn_models import GCNEdgeBased
from tools.evaluation_metric import *
from tools.cluster_functions import *


writer = SummaryWriter()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_dataset_ids = [1130025, 1195075, 1195448, 1232164, 1268839, 1292085,\
                    1354437, 1422331, 1422429, 1599988, 1631506, 1631582, 1725139,\
                    1725272, 196589, 264569, 447649, 5320, 581141, 581180, 795802,\
                    796175, 94638, 95289]

test_dataset_ids = [1079897, 1232423, 1599902, 196078]

val_dataset_ids = [1104787, 1387186, 388476, 65777]

def filterer(df):
    return df.loc[df['redshiftstar']<2].copy()

feature_columns = ['estar', 'jrstar', 'jzstar', 'jphistar', 'rstar', 'vstar', 'vxstar', 'vystar', 'vzstar', 'vrstar', 'vphistar', 'phistar', 'zstar']
position_columns = ['xstar', 'ystar', 'zstar']
data_transforms = T.Compose(transforms=[T.KNNGraph(k=300, force_undirected=True), T.GDC(sparsification_kwargs={'avg_degree':300, 'method':'threshold'})]) # 
train_dataset = NormalCaterpillarDataset('../data/caterpillar', '0', feature_columns, position_columns, use_dataset_ids=train_dataset_ids, data_filter=filterer, repeat=10, label_column='cluster_id', transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)  # it's already pre-shuffled. We can't do shuffling here because it must generate things in sequence.
val_dataset = NormalCaterpillarDataset('../data/caterpillar', '0', feature_columns, position_columns, use_dataset_ids=val_dataset_ids, data_filter=filterer, repeat=10, label_column='cluster_id', transform=data_transforms)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


EPOCH = 100

GCNEdgeBased_model = torch.load('weights/GCNEdgeBased_model300new/299.pth').to(device)


for i, data_batch in enumerate(val_loader):
	print(data_batch)
	data_batch = data_batch.to(device)
	with torch.no_grad():
		GCNEdgeBased_model.eval()
		edge_pred = GCNEdgeBased_model(data_batch)
		loss = GCNEdgeBased_model.loss(edge_pred, data_batch.edge_type)
	data_batch.edge_attr = edge_pred
	metric = None
	train_generator = T_Edge2Cluster(data_batch, gap=10, n_components=50, cluster_lr=0.0001, cluster_regularizer=0.01, epochs=4000, device=device)
	for j, (FX, loss) in enumerate(train_generator):
		print(torch.mean(torch.sum(FX, dim=0)), torch.std(torch.sum(FX, dim=0)), torch.max(torch.sum(FX, dim=0)))
		metric = ClusterEvalAll(FX.detach().cpu().numpy(), data_batch['y'].cpu().numpy())()
		writer.add_scalar('Edge2ClusterLoss', loss, 4000*i + j)
		writer.add_scalar('IoU_TP', metric['IoU_TP'], 4000*i + j)
		writer.add_scalar('IoU_recall', metric['IoU_recall'], 4000*i + j)
		writer.add_scalar('Mode_TP', metric['Mode_TP'], 4000*i + j)
		writer.add_scalar('Mode_recall', metric['Mode_recall'], 4000*i + j)
		writer.add_scalar('ModeProb_TP', metric['ModeProb_TP'], 4000*i + j)
		writer.add_scalar('ModeProb_recall', metric['ModeProb_recall'], 4000*i + j)
	print()
	print('final metric:')
	print(metric)
	print()

