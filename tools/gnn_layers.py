import numpy as np
import torch
from torch.nn import Linear, ReLU, MultiheadAttention, Module, ModuleList
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, softmax

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


class FeatureEncoder(Module):
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


class NodeATN(MessagePassing):
    def __init__(self, in_node_channels, in_edge_channels, out_node_channels):
        super().__init__(aggr='sum')  # sum aggregation
        self.self_map = Linear(in_node_channels, out_node_channels)
        self.pass_map = Linear(in_edge_channels, out_node_channels)
        self.attn_map = Linear(in_node_channels, in_edge_channels)
        self.relu = ReLU()

    def reset_parameters(self):
        self.self_map.reset_parameters()
        self.pass_map.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        x_out = self.self_map(x)
        row, col = edge_index # computing degree

        x_attn = self.attn_map(x)
        edge_alpha = torch.sum(edge_attr * x_attn[col], dim=-1)
        edge_alpha = softmax(edge_alpha, col)  # normalize

        message_received = self.propagate(edge_index, size=(x.size(0), x.size(0)), edge_attr=edge_attr * edge_alpha[...,None])
        message_received = self.pass_map(message_received) # act on edge features
        out = self.relu(x_out + message_received)  # ReLU(xW + sum_neighbor EW)

        return out

    def message(self, edge_attr):
        # edge_attr has shape [E, out_channels]
        return edge_attr


class NodeATNOrig(MessagePassing):
    def __init__(self, in_node_channels, in_edge_channels, out_node_channels, heads):
        super().__init__(aggr='sum')  # sum aggregation
        self.heads = heads
        self.node_attn_maps = ModuleList([Linear(in_node_channels, out_node_channels // heads) for i in range(heads)])
        self.edge_attn_maps = ModuleList([Linear(in_edge_channels, out_node_channels // heads) for i in range(heads)])
        self.combinator = Linear(out_node_channels * 2 // heads, 1)
        self.relu = ReLU()

    def reset_parameters(self):
        self.node_attn_maps.reset_parameters()
        self.edge_attn_maps.reset_parameters()
        self.combinator.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        row, col = edge_index # computing degree

        x_heads = [linear_map(x) for linear_map in self.node_attn_maps]
        e_heads = [linear_map(edge_attr) for linear_map in self.edge_attn_maps]

        e_alphas = [self.combinator(torch.cat([x_head[col], e_head], dim=-1)) for x_head, e_head in zip(x_heads, e_heads)]
        e_alphas = [softmax(e_alpha, col) for e_alpha in e_alphas]

        e_heads = [e_head * e_alpha for e_head, e_alpha in zip(e_heads, e_alphas)]

        messages_received = [self.propagate(edge_index, size=(x.size(0), x.size(0)), edge_attr=e_head) for e_head in e_heads]
        messages_received = [self.relu(message) for message in messages_received]

        out = torch.cat(messages_received, dim=-1)

        return out

    def message(self, edge_attr):
        # edge_attr has shape [E, out_channels]
        return edge_attr