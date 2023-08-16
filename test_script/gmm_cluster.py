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


writer = SummaryWriter()
device = torch.device('cpu') 

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

test_dataset = NormalCaterpillarDataset('../data/caterpillar', '0', feature_columns, position_columns, use_dataset_ids=test_dataset_ids, data_filter=filterer, repeat=10, label_column='cluster_id')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


def evaluate_all(n_components, loader):
	metrics = []
	for i, data_batch in enumerate(loader):
		print(data_batch)
		data_batch_train = data_batch.to(device)
		labels = C_GaussianMixture(data_batch_train['x'], n_components=n_components) 
		# more work needed here
		metric = ClusterEvalAll(labels, data_batch['y'].cpu().numpy())()
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
for n_components in [2,3,4,5,7,10,20,40,60,80,120,160,200]:
	metric = evaluate_all(n_components, test_loader)
	results[n_components] = metric

with open('../results/gaussian_mixture_full.json', 'w') as f:
	json.dump(results, f)
