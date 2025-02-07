"""Module with the definition of the GNN model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCN, summary, Sequential
from torch_geometric.nn.conv import GraphConv, NNConv, TransformerConv, PDNConv
from torch.nn import ReLU, ModuleList


class GraphConvNetwork(torch.nn.Module):
    """Simple Graph Convolutional Network model."""
    def __init__(self, input_dim, output_dim, hidden_dim=16, layers=2, p_drop=0.0):
        super().__init__()

        self.convs = ModuleList()
        self.convs.append(GraphConv(input_dim, hidden_dim, aggr='add'))
        for _ in range(layers - 2):
            self.convs.append(GraphConv(hidden_dim, hidden_dim, aggr='add'))
        self.convs.append(GraphConv(hidden_dim, output_dim, aggr='add'))

        self.drop = nn.Dropout(p_drop)

    def forward(self, data):
        """Forward pass of the model."""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.drop(x)
        x = self.convs[-1](x, edge_index, edge_attr)

        return x




# MARK: - Test the model
from samples import LinEqSample, DynamicGraphDataset # pylint: disable=all
# from torch_geometric.loader import DataLoader

if __name__ == '__main__':
    # Test the model
    options = {
        'n': 100,
        'rank': 5,
        'off_diagonal_abs_mean': 1.0,
        'symmetric': False,
        'uniform_range': (0.1,10.)
    }
    samples = LinEqSample(**options)

    dataset = DynamicGraphDataset(
        dataset_len=100,
        sample_function=samples.get_graph,
        max_error=1E-6,
        max_iter=1000,
        throw_error=True
    )
    # loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = GraphConvNetwork(dataset, hidden=16, layers=5)
    sample = dataset[0]
    result = model(sample)
    print(f'Expected: {sample.y.shape}')
    print(f'Result: {result.shape}')
    print(result)
