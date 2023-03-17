import sys
sys.path.append('..')

import torch
from torch.optim import SGD, Adam
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
import torch_geometric.transforms as T
from torch.utils.tensorboard import SummaryWriter
from tools.caterpillar_dataset import NormalCaterpillarDataset
from tools.gnn_models import GCNEdgeBased
from tools.evaluation_metric import *
from tools.neighbor_loader_fixer import add_indices


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
train_dataset = NormalCaterpillarDataset('../data/caterpillar', '0', feature_columns, position_columns, use_dataset_ids=train_dataset_ids, data_filter=filterer, repeat=10, label_column='cluster_id', transform=T.KNNGraph(k=1000, force_undirected=True))
val_dataset = NormalCaterpillarDataset('../data/caterpillar', '0', feature_columns, position_columns, use_dataset_ids=val_dataset_ids, data_filter=filterer, repeat=10, label_column='cluster_id', transform=T.KNNGraph(k=1000, force_undirected=True))



EPOCH = 100

GCNEdgeBased_model = GCNEdgeBased(len(feature_columns), regularizer=0).to(device)
GCNEdgeBased_optim = Adam(GCNEdgeBased_model.parameters(), lr=0.001, weight_decay=1e-5)

def train_one_batch(model, optim, data_batch, evaluate=False):
	model.train()
	optim.zero_grad()
	edge_pred = model(data_batch)
	edge_pred = edge_pred[data_batch.match_id]
	edge_type = data_batch.edge_type[data_batch.match_id]
	loss = model.loss(edge_pred, edge_type)
	loss.backward()
	optim.step()
	if evaluate:
		classification_metric = ClassificationAcc(torch.round(edge_pred.detach().cpu()).long(), edge_type.detach().cpu().long() ,2)
		return loss.cpu().item(), classification_metric
	return loss.cpu().item(), None


def evaluate_one_batch(model, data_batch):
	model.eval()
	with torch.no_grad():
		edge_pred = model(data_batch)
		edge_pred = edge_pred[data_batch.match_id]
		edge_type = data_batch.edge_type[data_batch.match_id]
		loss = model.loss(edge_pred, edge_type)
		classification_metric = ClassificationAcc(torch.round(edge_pred.detach().cpu()).long(), edge_type.detach().cpu().long() ,2)
		return loss.cpu().item(), classification_metric


for epoch in range(EPOCH):
	print('training begins...')
	for i, full_data in enumerate(train_dataset):
		evaluate_acc = (i%10)==0
		print(full_data.edge_index.shape[-1])
		data_loader = LinkNeighborLoader(full_data, num_neighbors=[256, 256], batch_size=8192, edge_label_index=full_data.edge_index, edge_label=full_data.edge_type, transform=add_indices)
		losses = []
		accs = []
		for data_batch in data_loader:
			loss, acc = train_one_batch(GCNEdgeBased_model, GCNEdgeBased_optim, data_batch.to(device), evaluate=evaluate_acc)
			losses.append(loss)
			if evaluate_acc:
				accs.append(acc())
		loss = sum(losses)/len(losses)
		print(loss)
		if evaluate_acc:
			acc = ClassificationAcc.aggregate(accs)
			print(acc)
		writer.add_scalar('Train/Loss', loss, epoch*len(train_dataset)+i)

	print('evaluation begins...')
	validation_accs = []
	losses = []
	for i, full_data in enumerate(val_dataset):
		print_acc = (i%10)==0
		data_loader = LinkNeighborLoader(full_data, num_neighbors=[256, 256], batch_size=8192, edge_label_index=full_data.edge_index, edge_label=full_data.edge_type, transform=add_indices)

		for data_batch in data_loader:
			loss, acc = evaluate_one_batch(GCNEdgeBased_model, data_batch.to(device))
			validation_accs.append(acc())
			losses.append(loss)
		if print_acc:
			print(loss)
			print(acc())

	validation_acc = ClassificationAcc.aggregate(validation_accs)
	avg_loss = sum(losses)/len(losses)
	print(avg_loss)
	print(validation_acc)
	writer.add_scalar('Val/Loss', avg_loss, epoch)
	writer.add_scalar('Val/EdgeRecall', validation_acc['recall'], epoch)
	writer.add_scalar('Val/Acc', validation_acc['accuracy'], epoch)

	torch.save(GCNEdgeBased_model, f'weights/GCNEdgeBased_model1000/{epoch}.pth')


