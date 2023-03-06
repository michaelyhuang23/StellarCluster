import torch
from torch import nn
import torch.nn.functional as F

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