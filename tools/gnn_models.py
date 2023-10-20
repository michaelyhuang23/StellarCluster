import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import sys
sys.path.append('..')
from tools.gnn_layers import NodeConv, EdgeConv, NodeATN, NodeATNOrig




class GCNEdgeBased(nn.Module): # non-overlapping
    def __init__(self, input_size, regularizer=0.1):
        super().__init__()
        self.input_size = input_size
        self.convN1 = NodeConv(input_size, input_size, 32)
        self.dropout1 = nn.Dropout(p=0.0)
        self.convE1 = EdgeConv(32, input_size, 32)
        self.convN2 = NodeConv(32, 32, 32)
        self.dropout2 = nn.Dropout(p=0.0)
        self.convE2 = EdgeConv(32, 32, 32)
        self.classifier = nn.Linear(32, 1)
        self.regularizer = regularizer

    def regularize(self, edge_pred):
        return -torch.mean((edge_pred-torch.mean(edge_pred))**4)**0.25 * self.regularizer

    def loss(self, edge_pred, edge_type, similar_weight=1):
        if edge_type.dtype != torch.float32:
            edge_type = edge_type.float()
        weight = torch.ones_like(edge_type)
        weight[edge_type > 0.5] *= similar_weight
        return F.binary_cross_entropy(edge_pred, edge_type, weight=weight) + self.regularize(edge_pred)

    def forward(self, data):
        X, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if edge_attr is None or len(edge_attr) == 0:
            edge_attr = X[edge_index[1]] - X[edge_index[0]]  # should we use abs? 
        else:
            edge_attr = (X[edge_index[1]] - X[edge_index[0]]) / edge_attr[..., None]
        X = torch.zeros_like(X)
        X = self.convN1(X, edge_index, edge_attr)
        X = self.dropout1(X)
        edge_attr = self.convE1(X, edge_index, edge_attr)
        X = self.convN2(X, edge_index, edge_attr)
        X = self.dropout2(X)
        edge_attr = self.convE2(X, edge_index, edge_attr)

        edge_pred = self.classifier(edge_attr)
        edge_pred = torch.sigmoid(edge_pred)[:,0]
        return edge_pred


class GANEdgeBased(nn.Module): # non-overlapping
    def __init__(self, input_size, regularizer=0.1):
        super().__init__()
        self.input_size = input_size
        self.attnN1 = NodeATN(input_size, input_size, 32)
        self.convE1 = EdgeConv(32, input_size, 32)
        self.attnN2 = NodeATN(32, 32, 32)
        self.convE2 = EdgeConv(32, 32, 32)
        self.classifier = nn.Linear(32, 1)
        self.regularizer = regularizer

    def regularize(self, edge_pred):
        return -torch.mean((edge_pred-torch.mean(edge_pred))**4)**0.25 * self.regularizer

    def loss(self, edge_pred, edge_type, similar_weight=1):
        if edge_type.dtype != torch.float32:
            edge_type = edge_type.float()
        weight = torch.ones_like(edge_type)
        weight[edge_type > 0.5] *= similar_weight
        return F.binary_cross_entropy(edge_pred, edge_type, weight=weight) + self.regularize(edge_pred)

    def forward(self, data):
        X, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if edge_attr is None or len(edge_attr) == 0:
            edge_attr = X[edge_index[1]] - X[edge_index[0]]  # should we use abs? 
        else:
            edge_attr = (X[edge_index[1]] - X[edge_index[0]]) / edge_attr[..., None]
        X = torch.zeros_like(X)
        X = self.attnN1(X, edge_index, edge_attr)
        edge_attr = self.convE1(X, edge_index, edge_attr)
        X = self.attnN2(X, edge_index, edge_attr)
        edge_attr = self.convE2(X, edge_index, edge_attr)

        edge_pred = self.classifier(edge_attr)
        edge_pred = torch.sigmoid(edge_pred)[:,0]
        return edge_pred


class GANOrigEdgeBased(nn.Module): # non-overlapping
    def __init__(self, input_size, regularizer=0.1):
        super().__init__()
        self.input_size = input_size
        self.attnN1 = NodeATNOrig(input_size, input_size, 32, 4)
        self.convE1 = EdgeConv(32, input_size, 32)
        self.attnN2 = NodeATNOrig(32, 32, 32, 4)
        self.convE2 = EdgeConv(32, 32, 32)
        self.classifier = nn.Linear(32, 1)
        self.regularizer = regularizer

    def regularize(self, edge_pred):
        return -torch.mean((edge_pred-torch.mean(edge_pred))**4)**0.25 * self.regularizer

    def loss(self, edge_pred, edge_type, similar_weight=1):
        if edge_type.dtype != torch.float32:
            edge_type = edge_type.float()
        weight = torch.ones_like(edge_type)
        weight[edge_type > 0.5] *= similar_weight
        return F.binary_cross_entropy(edge_pred, edge_type, weight=weight) + self.regularize(edge_pred)

    def forward(self, data):
        X, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if edge_attr is None or len(edge_attr) == 0:
            edge_attr = X[edge_index[1]] - X[edge_index[0]]  # should we use abs? 
        else:
            edge_attr = (X[edge_index[1]] - X[edge_index[0]]) / edge_attr[..., None]
        X = torch.zeros_like(X)
        X = self.attnN1(X, edge_index, edge_attr)
        edge_attr = self.convE1(X, edge_index, edge_attr)
        X = self.attnN2(X, edge_index, edge_attr)
        edge_attr = self.convE2(X, edge_index, edge_attr)

        edge_pred = self.classifier(edge_attr)
        edge_pred = torch.sigmoid(edge_pred)[:,0]
        return edge_pred


class GCNEdge2Cluster_prob(nn.Module):
    def __init__(self, input_size, num_cluster=30, regularizer=0.01):
        super().__init__()
        self.conv1 = GCNConv(input_size, 256)
        self.linear1 = nn.Linear(input_size, 256)
        self.conv2 = GCNConv(256, num_cluster)
        self.linear2 = nn.Linear(256, num_cluster)
        self.num_cluster = num_cluster
        self.regularizer = regularizer

    def loss(self, gen_edge_prob, edge_prob):
        return F.binary_cross_entropy(gen_edge_prob, edge_prob)
        #return torch.mean((gen_edge_prob - edge_prob)**2) 

    def forward(self, data):
        X, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        X = F.relu(self.conv1(X, edge_index, edge_attr) + self.linear1(X))
        X = self.conv2(X, edge_index, edge_attr) + self.linear2(X)
        FX = torch.clamp(F.softmax(X, dim=-1), min=1e-9, max=1-1e-9) # change to softmax if not doing overlapping clusters
        FF = torch.clamp(torch.sum(FX[edge_index[0], :-1] * FX[edge_index[1], :-1], dim=-1), min=1e-9, max=1-1e-9)
        NFX = torch.log(1-FX**2)
        pregularize = -torch.sum(torch.log(1.0001-torch.exp(torch.sum(NFX, dim=0))), dim=0)
        #loss = torch.mean((FF - edge_attr)**2) 
        loss = self.loss(FF, edge_attr)
        return FX, loss + self.regularizer * pregularize


class GCNEdge2Cluster_modularity(nn.Module):
    def __init__(self, input_size, num_cluster=30, regularizer=0.01):
        super().__init__()
        self.dropout0 = nn.Dropout(p=0.3)
        self.conv1 = GCNConv(input_size, 256)
        self.linear1 = nn.Linear(input_size, 256)
        self.dropout1 = nn.Dropout(p=0.3)
        self.conv2 = GCNConv(256, num_cluster)
        self.linear2 = nn.Linear(256, num_cluster)
        self.num_cluster = num_cluster
        self.regularizer = regularizer

    def loss(self, FX, edge_index, edge_attr):
        m = torch.sum(edge_attr)
        degree = torch.zeros((len(FX)), device=edge_attr.device)
        degree = degree.index_add(0, edge_index[0], edge_attr)
        connectivity = torch.sum(edge_attr * torch.sum(FX[edge_index[0], :-1] * FX[edge_index[1], :-1], dim=-1), dim=0)  # sparse tensor product
        avg_connectivity = torch.dot(torch.sum(FX * degree[...,None], dim=0)[:-1], torch.sum(FX * degree[...,None], dim=0)[:-1]) / (2 * m)
        modularity = - 1/(2*m) * (connectivity - avg_connectivity)
        return modularity

    def regularize(self, FX):
        return torch.sqrt(torch.sum(torch.sum(FX, dim=0) ** 2, dim=-1) + 1e-9) * FX.shape[-1]**0.5 / FX.shape[0] - 1

    def forward(self, data):
        X, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        X = self.dropout0(X)
        X = self.dropout1(F.relu(self.conv1(X, edge_index, edge_attr) + self.linear1(X)))
        X = self.conv2(X, edge_index, edge_attr) + self.linear2(X)
        FX = torch.clamp(F.softmax(X, dim=-1), min=1e-9, max=1-1e-9) # change to softmax if not doing overlapping clusters
        loss = self.loss(FX, edge_index, edge_attr)
        pregularize = self.regularize(FX)
        return FX, loss + self.regularizer * pregularize


class FeedForwardProjector(nn.Module):
    def __init__(self, input_size, hidden_sizes, n_components=5):
        super().__init__()
        self.input_size = input_size
        self.n_components = n_components
        self.layers = nn.ModuleList()
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(input_size, hidden_size))
            self.layers.append(nn.ReLU())
            input_size = hidden_size
        self.layers.append(nn.Linear(input_size, n_components))

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        X = F.softmax(X, dim=-1)
        return X

    def edge_pred_loss(self, edge_index, edge_pred, FX):
        FX = torch.sum(FX[edge_index[0]] * FX[edge_index[1]], dim=-1)
        print(FX)
        print(edge_pred)
        return F.cross_entropy(FX, edge_pred)

    def edge_pred_acc(self, edge_index, edge_pred, FX):
        FX = torch.sum(FX[edge_index[0]] * FX[edge_index[1]], dim=-1)
        return torch.mean(((FX>0.5) == (edge_pred>0.5)).float())


