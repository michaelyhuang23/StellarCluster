import sys
sys.path.append('..')
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from tools.neighbor_loader_fixer import add_indices

data = Planetoid('data/', name='Cora')[0]



loader = LinkNeighborLoader(
    data,
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors=[2] * 2,
    # Use a batch size of 128 for sampling training nodes
    batch_size=5,
    edge_label_index=data.edge_index,
    transform=add_indices
)

print(len(data.x))


for sampled_data in loader:
    print(sampled_data)
    print(sampled_data.edge_index[:, sampled_data.match_id])
    print(sampled_data.edge_label_index)





