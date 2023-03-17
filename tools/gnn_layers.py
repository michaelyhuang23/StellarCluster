import torch
from torch.nn import Linear, ReLU
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

