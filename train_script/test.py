import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader

data = Planetoid('data/', name='Cora')[0]



loader = LinkNeighborLoader(
    data,
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors=[10] * 2,
    # Use a batch size of 128 for sampling training nodes
    batch_size=30,
    edge_label_index=data.edge_index
)

print(len(data.edge_index[0]))

for sampled_data in loader:
	print(sampled_data)
	print(sampled_data.input_id)
	print(sampled_data.edge_index)
	print(sampled_data.edge_label_index)

