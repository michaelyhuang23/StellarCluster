import torch

def add_indices(data_batch): # complicated tensor ops to map edges in edge_label_index back to indices of edges in edge_index
	n = len(data_batch.x)
	edge_label_index = data_batch.edge_label_index
	edge_index = data_batch.edge_index

	edge_label_index_order = edge_label_index[0] * n + edge_label_index[1]
	edge_index_order = edge_index[0] * n + edge_index[1]

	edge_index_order = torch.concatenate([edge_index_order, edge_label_index_order])
	edge_index_order, indices = torch.unique(edge_index_order, sorted=True, return_inverse=True)
	print(indices)

	edge_index = torch.concatenate([edge_index, edge_label_index], dim=-1)
	new_edge_index = torch.zeros((2, len(edge_index_order))).long()
	new_edge_index[:, indices] = edge_index
	data_batch.edge_index = new_edge_index

	if 'edge_type' in data_batch.keys:
		edge_type = torch.concatenate([data_batch.edge_type, data_batch.edge_label])  # must have valid edge_labels
		new_edge_type = torch.zeros((len(edge_index_order))).long()
		new_edge_type[indices] = edge_type
		data_batch.edge_type = new_edge_type

	match_id = torch.searchsorted(edge_index_order, edge_label_index_order)
	data_batch.match_id = match_id

	return data_batch

