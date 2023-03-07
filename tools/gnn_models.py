import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import sys
sys.path.append('..')
from tools.gnn_layers import NodeConv, EdgeConv




class GCNEdgeBased(nn.Module): # non-overlapping
    def __init__(self, input_size, similar_weight=1, regularizer=0.1):
        super().__init__()
        self.input_size = input_size
        self.convN1 = NodeConv(input_size, input_size, 32)
        self.dropout1 = nn.Dropout(p=0.0)
        self.convE1 = EdgeConv(32, input_size, 32)
        self.convN2 = NodeConv(32, 32, 32)
        self.dropout2 = nn.Dropout(p=0.0)
        self.convE2 = EdgeConv(32, 32, 32)
        self.classifier = nn.Linear(32, 1)
        self.similar_weight = similar_weight
        self.regularizer = regularizer

    def forward(self, data):
        X, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if edge_attr is None or len(edge_attr) == 0:
            edge_attr = X[edge_index[1]] - X[edge_index[0]]  # should we use abs? 
        X = torch.zeros_like(X) 
        X = self.convN1(X, edge_index, edge_attr)
        X = self.dropout1(X)
        edge_attr = self.convE1(X, edge_index, edge_attr)
        X = self.convN2(X, edge_index, edge_attr)
        X = self.dropout2(X)
        edge_attr = self.convE2(X, edge_index, edge_attr)
        edge_pred = self.classifier(edge_attr)
        edge_pred = torch.sigmoid(edge_pred)[:,0]
        #print(torch.mean(edge_pred).item(), torch.std(edge_pred).item())
        loss_regularze = -torch.mean((edge_pred-torch.mean(edge_pred))**4)**0.25
        #print(loss_regularze.item())

        weights = torch.ones_like(data.edge_type).float()
        weights[edge_pred>0.5] *= self.similar_weight
        loss = F.binary_cross_entropy(edge_pred, data.edge_type.float(), weight=weights)
        return edge_pred, loss + loss_regularze * self.regularizer


class GCNEdge2Cluster(nn.Module):
    def __init__(self, input_size, num_cluster=30, regularizer=0.01):
        super().__init__()
        self.conv1 = GCNConv(input_size, 32)
        self.dropout1 = nn.Dropout(p=0)
        self.conv2 = GCNConv(32, num_cluster)
        self.num_cluster = num_cluster
        self.regularizer = regularizer

    def forward(self, data, edge_pred):
        X, edge_index = data.x, data.edge_index
        X = F.relu(self.conv1(X, edge_index))
        X = self.dropout1(X)
        X = self.conv2(X, edge_index)
        FX = F.softmax(X, dim=-1) # change to softmax if not doing overlapping clusters
        FF = torch.sum(FX[edge_index[0]] * FX[edge_index[1]], dim=-1)
        NFX = torch.log(1-FX**2)
        pregularize = -torch.sum(torch.log(1.0001-torch.exp(torch.sum(NFX, dim=0))), dim=0)
        loss = torch.mean((FF - edge_pred)**2) + self.regularizer * pregularize
        return FX, loss

