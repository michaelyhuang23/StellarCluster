import sys
sys.path.append('..')

import json

import torch
from torch.optim import SGD, Adam
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch.utils.tensorboard import SummaryWriter
from tools.caterpillar_dataset import NormalCaterpillarDataset
from tools.gnn_models import GCNEdgeBased
from tools.evaluation_metric import *
from tools.cluster_functions import *
from scipy.sparse import csr_array, csr_matrix
from torch_geometric.utils import remove_self_loops
from torcheval.metrics.functional import binary_accuracy

writer = SummaryWriter()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)
train_dataset_ids = [1130025, 1195075, 1195448, 1232164, 1268839, 1292085,\
                    1354437, 1422331, 1422429, 1599988, 1631506, 1631582, 1725139,\
                    1725272, 196589, 264569, 447649, 5320, 581141, 581180, 795802,\
                    796175, 94638, 95289]

test_dataset_ids = [1079897, 1232423, 1599902, 196078]

val_dataset_ids = [1104787, 1387186, 388476, 65777]

def filterer(df):
    return df.loc[df['redshiftstar']<2].copy()

feature_columns = ['estar', 'jrstar', 'jzstar', 'jphistar', 'rstar', 'vstar', 'vxstar', 'vystar', 'vzstar', 'vrstar', 'vphistar', 'phistar', 'zstar']
#feature_columns = ['estar', 'jrstar', 'jzstar', 'jphistar', 'vstar', 'vzstar', 'vrstar', 'vphistar']
position_columns = ['xstar', 'ystar', 'zstar']
data_transforms = T.Compose(transforms=[T.KNNGraph(k=300, force_undirected=True, loop=False), T.GDC(normalization_out='sym', self_loop_weight=None, sparsification_kwargs={'avg_degree':300, 'method':'threshold'})]) # 

test_dataset = NormalCaterpillarDataset('../data/caterpillar', '0', feature_columns, position_columns, use_dataset_ids=test_dataset_ids, data_filter=filterer, repeat=10, label_column='cluster_id', transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


model = GCNEdgeBased(len(feature_columns), regularizer=0).to(device)
model.load_state_dict(torch.load('../train_script/weights/GCNEdgeBased_model300_Aug_24/299.pth', map_location=device)['model_state_dict'])



def evaluate_all(n_components, loader, model):
	metrics = []
	for i, data_batch in enumerate(loader):
		data_batch_train = data_batch.to(device)
		with torch.no_grad():
			model.eval()
			edge_pred = model(data_batch_train)
			metric = ClassificationAcc(torch.round(edge_pred).long(), data_batch_train.edge_type.long(), 2)
			print(f'acc: {metric()}')
			print(torch.sum((edge_pred>0.5).long())/len(edge_pred))
			loss = model.loss(edge_pred, data_batch_train.edge_type)
			print(loss)
		#adj = csr_matrix((data_batch_train.edge_attr.numpy(), (data_batch_train.edge_index[0].numpy(), data_batch_train.edge_index[1].numpy())), shape=(len(data_batch_train.x), len(data_batch_train.x)))
		adj = torch.sparse_coo_tensor(data_batch_train.edge_index.cpu(), edge_pred.cpu(), (len(data_batch_train.x), len(data_batch_train.x))).to_dense()
		adj = (adj + adj.T)/2
		FX = C_Spectral(adj, n_components=n_components)

		metric = ClusterEvalAll(FX, data_batch['y'].cpu().numpy())()
		writer.add_scalar('IoU_TP', metric['IoU_TP'], 4000*i)
		writer.add_scalar('IoU_recall', metric['IoU_recall'], 4000*i)
		writer.add_scalar('Mode_TP', metric['Mode_TP'], 4000*i)
		writer.add_scalar('Mode_recall', metric['Mode_recall'], 4000*i)
		writer.add_scalar('ModeProb_TP', metric['ModeProb_TP'], 4000*i)
		writer.add_scalar('ModeProb_recall', metric['ModeProb_recall'], 4000*i)

		
		print()
		print('current metric:')
		print(metric)
		print()
		metrics.append(metric)
	f_metric = ClusterEvalAll.aggregate(metrics)
	return f_metric

results = {}
for n_components in [5, 10, 20, 30, 40, 50, 60, 70, 80, 100]:
	metric = evaluate_all(n_components, test_loader, model)
	results[n_components] = metric

with open('../results/SpectralEdge2Cluster_test_GCN_retrain_299.json', 'w') as f:
	json.dump(results, f)
