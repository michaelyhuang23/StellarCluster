import numpy as np
import torch
from torch.nn import Linear, ReLU, MultiheadAttention
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

class NodeConv(MessagePassing):
    def __init__(self, in_node_channels, in_edge_channels, out_node_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.self_map = Linear(in_node_channels, out_node_channels)
        self.pass_map = Linear(in_edge_channels, out_node_channels)
        self.relu = ReLU()

    def reset_parameters(self):
    	self.self_map.reset_parameters()
    	self.pass_map.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        x = self.self_map(x)

        row, col = edge_index # computing degree
        deg = degree(col, x.size(0), dtype=x.dtype) + 1     # add 1 in place of self-loop
        deg_inv = deg.pow(-1)

        edge_attr = self.pass_map(edge_attr)
        message_received = self.propagate(edge_index, size=(x.size(0), x.size(0)), edge_attr=edge_attr) * deg_inv[..., None]

        out = self.relu(x + message_received)  # ReLU(xW + sum_neighbor EW)

        return out

    def message(self, edge_attr):
        # edge_attr has shape [E, out_channels]
        return edge_attr


class EdgeConv(MessagePassing):
    def __init__(self, in_node_channels, in_edge_channels, out_edge_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.self_map = Linear(in_edge_channels, out_edge_channels)
        self.pass_map = Linear(in_node_channels*2, out_edge_channels)
        self.relu = ReLU()

    def reset_parameters(self):
        self.self_map.reset_parameters()
        self.pass_map.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        edge_attr = self.self_map(edge_attr)

        row, col = edge_index

        message_received = self.pass_map(torch.concat([x[row], x[col]], dim=-1)) # from the two nodes

        out = self.relu(edge_attr + message_received)  # ReLU(EW + [X,X]W)

        return out




class FeatureEncoder(nn.Module):
    def __init__(self, flen=13, d=13, n=10000):
        self.n = n
        self.d = d
        self.flen = flen
        freq_method = []
        for i in range(self.d):
            k = (i//2)*2
            inv_freq = 1/(self.n ** (k/self.d))
            if i % 2 == 0:
                freq_method.append((inv_freq, True))
            else:
                freq_method.append((inv_freq, False))

        freq_methods = [np.random.permutation(freq_method) for i in range(self.flen)]
        self.inv_freqs = torch.tensor([[inv_freq for (inv_freq, method) in freq_method] for freq_method in freq_methods]).float()
        self.methods = torch.tensor([[method for (inv_freq, method) in freq_method] for freq_method in freq_methods]).bool()


    def encode_feature(self, features, sin_inv_freqs, cos_inv_freqs): # feature is n by 13, # sin_inv_ferqs is 13 by d
        sin_encodings = torch.sum(torch.sin(torch.einsum('ij,jk->ijk', features, sin_inv_freqs)), 1) # collapse j dim
        cos_encodings = torch.sum(torch.cos(torch.einsum('ij,jk->ijk', features, cos_inv_freqs)), 1) # collapse j dim
        return None

    def encode_feature_column(self, features, inv_freqs, methods): # feature is n by 13, inv_freqs is 13
        methods = methods.float()
        sinusoidal_features = torch.sin(features * inv_freqs[None, ...]) * methods[None, ...] + torch.cos(features * inv_freqs[None, ...]) * (1-methods[None, ...])
        return torch.sum(sinusoidal_features, dim=-1)

    def forward(self, features):
        encodings = []
        for i in range(self.d):
            column_encoding = self.encode_feature_column(features, self.inv_freqs[:,i], self.methods[:,i])
            encodings.append(column_encoding)
        encoded_features = torch.stack(encodings, dim=1) # should be n by self.d
        return encoded_features


