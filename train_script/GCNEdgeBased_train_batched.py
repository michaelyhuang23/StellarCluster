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
train_dataset = NormalCaterpillarDataset('../data/caterpillar', '0', feature_columns, position_columns, use_dataset_ids=train_dataset_ids, data_filter=filterer, repeat=10, label_column='cluster_id', transform=T.KNNGraph(k=5, force_undirected=True))
val_dataset = NormalCaterpillarDataset('../data/caterpillar', '0', feature_columns, position_columns, use_dataset_ids=val_dataset_ids, data_filter=filterer, repeat=10, label_column='cluster_id', transform=T.KNNGraph(k=5, force_undirected=True))



EPOCH = 100

GCNEdgeBased_model = GCNEdgeBased(len(feature_columns), regularizer=0).to(device)
GCNEdgeBased_optim = Adam(GCNEdgeBased_model.parameters(), lr=0.001, weight_decay=1e-5)

def train_one_batch(model, optim, data_batch, evaluate=False):
	model.train()
	optim.zero_grad()
	pred, loss = model(data_batch)
	loss.backward()
	optim.step()
	if evaluate:
		classification_metric = ClassificationAcc(torch.round(pred.detach().cpu()).long(), data_batch.edge_type.detach().cpu().long() ,2)
		return loss.cpu().item(), classification_metric
	return loss.cpu().item(), None


def evaluate_one_batch(model, data_batch):
	model.eval()
	pred, loss = model(data_batch)
	classification_metric = ClassificationAcc(torch.round(pred.detach().cpu()).long(), data_batch.edge_type.detach().cpu().long() ,2)
	return loss.cpu().item(), classification_metric


for epoch in range(EPOCH):
	print('training begins...')
	for i, full_data in enumerate(train_dataset):
		evaluate_acc = (i%10)==0
		data_loader = LinkNeighborLoader(full_data, num_neighbors=[10, 10], batch_size=128)

		data_loader_iter = iter(data_loader)
		next(data_loader_iter)
		mini_data = next(data_loader_iter)
		print(mini_data)
		print(mini_data.input_id)
		print(mini_data.edge_label_index)
		print(mini_data.edge_index)
		print(mini_data.keys)


		for data_batch in data_loader:
			loss, acc = train_one_batch(GCNEdgeBased_model, GCNEdgeBased_optim, data_batch.to(device), evaluate=evaluate_acc)
			print(loss)

		if evaluate_acc:
			print(acc())
		writer.add_scalar('Train/Loss', loss, epoch*len(train_dataset)+i)

	print('evaluation begins...')
	validation_accs = []
	losses = []
	for i, data_batch in enumerate(val_loader):
		loss, acc = evaluate_one_batch(GCNEdgeBased_model, data_batch.to(device))
		validation_accs.append(acc())
		losses.append(loss)
		if i%10 == 0:
			print(loss)
			print(acc())

	validation_acc = ClassificationAcc.aggregate(validation_accs)
	avg_loss = sum(losses)/len(losses)
	print(avg_loss)
	print(validation_acc)
	writer.add_scalar('Val/Loss', avg_loss, epoch)
	writer.add_scalar('Val/EdgeRecall', validation_acc['recall'], epoch)
	writer.add_scalar('Val/Acc', validation_acc['accuracy'], epoch)

	torch.save(GCNEdgeBased_model, f'GCNEdgeBased_model1000/{epoch}.pth')


