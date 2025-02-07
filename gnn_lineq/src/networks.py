"""Module with the definition of the GNN model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCN, summary, Sequential
from torch_geometric.nn.conv import GraphConv, NNConv, TransformerConv, PDNConv
from torch.nn import ReLU, ModuleList


class ActivationFunction(nn.Module):
    """Activation function."""
    def __init__(self, activation='ReLU', **kwargs):
        super().__init__()

        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(**kwargs) # negative_slope=0.01
        elif activation == 'ELU':
            self.activation = nn.ELU(**kwargs) # alpha=1.0
        else:
            msg = f'Unknown activation function: {activation}.'
            msg +=' Use ReLU, Sigmoid, Tanh, LeakyReLU or ELU.'
            raise ValueError(msg)

    def forward(self, x):
        """Forward pass of the activation function."""
        return self.activation(x)

class SimpleMLP(torch.nn.Module):
    """
    Simple Multi-Layer Perceptron model.

    Parameters:
    -----------
    - input_dim (int): Dimension of the input.
    - output_dim (int): Dimension of the output.
    - hidden_dim (int): Dimension of the hidden layer. If None, model has a single
        layer followed by an activation function. Else, a linear layer is added after
        the activation function. Default is None.
    - activation (str): Activation function to use. Default is 'ReLU'.
    - **kwargs: Additional arguments for the activation function.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=None, activation='ReLU', **kwargs):
        super().__init__()

        activation = ActivationFunction(activation, **kwargs)

        if hidden_dim is None:
            self.layer = Sequential('x', [
                (torch.nn.Linear(input_dim, output_dim),'x -> x'),
                (activation,'x -> x'),
            ])
        else:
            self.layer = Sequential('x', [
                (torch.nn.Linear(input_dim, hidden_dim),'x -> x'),
                (activation,'x -> x'),
                (torch.nn.Linear(hidden_dim, output_dim),'x -> x')
            ])

    def forward(self, x):
        """Forward pass of the model."""
        return self.layer(x)


class GraphConvNetwork(torch.nn.Module):
    """Simple Graph Convolutional Network model."""
    def __init__(self, input_dim, output_dim, hidden_dim=16, layers=2, p_drop=0.0, activation='ReLU', **kwargs):
        super().__init__()

        self.convs = ModuleList()
        self.convs.append(GraphConv(input_dim, hidden_dim, aggr='add'))
        for _ in range(layers - 2):
            self.convs.append(GraphConv(hidden_dim, hidden_dim, aggr='add'))
        self.convs.append(GraphConv(hidden_dim, output_dim, aggr='add'))

        self.drop = nn.Dropout(p_drop)
        self.activation = ActivationFunction(activation, **kwargs)

    def forward(self, data):
        """Forward pass of the model."""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr)
            x = self.activation(x)
            x = self.drop(x)
        x = self.convs[-1](x, edge_index, edge_attr)

        return x


# class NNConvLayer(torch.nn.Module):
#     """Single NNConv layer."""
#     def __init__(self, input_dims, output_dims, hidden_dims, p_drop=0.0):
#         super().__init__()

#         edge_mlp = SimpleMLP(input_dims[1], hidden_dims[1], hidden_dims[0] * output_dims[0])
#         self.edge_mlps.append(GenericMLP(dataset.num_edge_features, hidden_edges, hidden_nodes * dataset.num_features))

#         self.conv = NNConv(hidden_nodes, hidden_nodes, nn=edge_mlp, aggr='add'))
#         self.drop = nn.Dropout(p_drop)


# class NNConvNetwork(torch.nn.Module):
#     """NNConv Network model."""
#     def __init__(self, input_dim, output_dim, hidden_nodes=16, hidden_edges=16, layers=2, p_drop=0.0):
#         super().__init__()

#         self.edge_mlps = ModuleList()
#         self.edge_mlps.append(GenericMLP(dataset.num_edge_features, hidden_edges, dataset.num_node_features * hidden_nodes))
#         for _ in range(layers - 2):
#             self.edge_mlps.append(GenericMLP(dataset.num_edge_features, hidden_edges, hidden_nodes * hidden_nodes))
#         self.edge_mlps.append(GenericMLP(dataset.num_edge_features, hidden_edges, hidden_nodes * dataset.num_features))

#         self.convs = ModuleList()
#         self.convs.append(NNConv(dataset.num_node_features, hidden_nodes, nn=self.edge_mlps[0], aggr='add'))
#         for i in range(layers - 2):
#             self.convs.append(NNConv(hidden_nodes, hidden_nodes, nn=self.edge_mlps[i+1], aggr='add'))
#         self.convs.append(NNConv(hidden_nodes, dataset.num_features, nn=self.edge_mlps[-1], aggr='add'))

#         self.use_dropout = use_dropout

#     def forward(self, data):
#         x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

#         for conv in self.convs[:-1]:
#             x = conv(x, edge_index, edge_attr)
#             x = F.relu(x)
#             if self.use_dropout:
#                 x = F.dropout(x, training=self.training)
#         x = self.convs[-1](x, edge_index, edge_attr)

#         return x


if __name__ == '__main__':
    print(__doc__)
