import sys
sys.path.append('..')
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from tools.neighbor_loader_fixer import add_indices


x = torch.tensor([0,1,3,1,0,2,0])
y = torch.tensor([1,2,3,4,5,6,7])

z = torch.zeros((4)).long()
z = z.index_add(0, x, y)

print(z)