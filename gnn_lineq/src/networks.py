# pylint: disable=line-too-long, too-many-arguments, unused-import
"""Module with the definition of the GNN model."""

import torch
from torch import nn
from torch.nn import ModuleList
# import torch.nn.functional as F

from torch_geometric.nn import GCN, summary, Sequential
from torch_geometric.nn.conv import GraphConv, NNConv, TransformerConv, PDNConv


class ActivationFunction(nn.Module):
    """Activation function."""
    def __init__(self, activation=None, **kwargs):
        super().__init__()

        if activation is None:
            self.activation = nn.Identity()
        elif not isinstance(activation, str):
            raise ValueError('Activation function must be a string or None.')
        elif activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        elif activation.lower() == 'leakyrelu':
            self.activation = nn.LeakyReLU(**kwargs) # negative_slope=0.01
        elif activation.lower() == 'elu':
            self.activation = nn.ELU(**kwargs) # alpha=1.0
        else:
            msg = f'Unknown activation function: {activation}.'
            msg +=" Use `None`, 'ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU' or 'ELU'."
            raise ValueError(msg)

    def forward(self, x):
        """Forward pass of the activation function."""
        return self.activation(x)


class MLP(torch.nn.Module):
    """
    Simple Multi-Layer Perceptron model.

    Parameters:
    -----------
    - dims (list of int): Dimension of the input, followed by the dimensions
        of the outputs of each layer.
    - p_drop (float): Dropout probability. Default is 0.0.
    - add_layer_norm (bool): If True, a layer normalization is added after
        the last layer. Default is False.
    - final_activation (bool): If True, an activation is added after the final
        layer. Default is False.
    - activation (str): Activation function to use. Default is `None`.
    - **kwargs: Additional arguments for the activation function.
    """
    def __init__(self, dims, p_drop=0.0, add_layer_norm=False, final_activation=False, activation=None, **kwargs):
        super().__init__()
        if len(dims) < 2:
            raise ValueError('At least two dimensions are required.')

        self.layers = ModuleList()
        activation = ActivationFunction(activation, **kwargs)

        for i in range(len(dims) - 1):
            self.layers.append(torch.nn.Linear(dims[i], dims[i+1]))
            if p_drop > 0.0:
                self.layers.append(nn.Dropout(p_drop))
            if i < len(dims) - 2 or final_activation:
                self.layers.append(activation)
        if add_layer_norm:
            self.layers.append(nn.LayerNorm(dims[-1]))

    def forward(self, x):
        """Forward pass of the model."""
        for layer in self.layers:
            x = layer(x)
        return x


class EdgeMLP(torch.nn.Module):
    """
    Edge Attribute MLP.

    Parameters:
    -----------
    - dims (list of int): Dimensions of the input (nodes, edges).
    - output_dims (list of int): Dimensions of the outputs of each layer.
    - p_drop (float): Dropout probability. Default is 0.0.
    - add_layer_norm (bool): If True, a layer normalization is added after
        the last layer. Default is False.
    - activation (str): Activation function to use. Default is `None`.
    - **kwargs: Additional arguments for the activation function.
    """
    def __init__(self, dims, output_dims, p_drop=0.0, add_layer_norm=False, activation=None, **kwargs):
        super().__init__()

        self.mlp = MLP(
            dims=[2*dims[0]+dims[1]] + output_dims,
            p_drop=p_drop,
            add_layer_norm=add_layer_norm,
            activation=activation,
            **kwargs)

    def forward(self, x, edge_index, edge_attr):
        """Forward pass of the layer."""
        e = torch.cat([x[edge_index[0]], x[edge_index[1]], edge_attr], dim=-1)
        e = self.mlp(e)
        return e


class NNConvLayer(torch.nn.Module):
    """
    Single NNConv layer + Edge Attribute MLP.

    Parameters:
    -----------
    - dims (tuple): Dimensions of the input and output (nodes, edges). Assumes
        output dimensions are equal to the input dimensions.
    - edge_mlp_layers (int): Number of layers for the edge MLP. Default is 2.
    - p_drop (float): Dropout probability. Default is 0.0.
    - add_layer_norm (bool): If True, a layer normalization is added after
        the last layer. Default is False.
    - nn_activation (str): Activation function to use for the nn. Default is `None`.
    - **kwargs: Additional arguments for the activation function
    """
    def __init__(self, dims, edge_mlp_layers=2, p_drop=0.0, add_layer_norm=False, nn_activation=None, **kwargs):
        super().__init__()

        self.edge_attr_mlp = MLP(
            dims=[2*dims[0]+dims[1]] + edge_mlp_layers*[dims[1]],
            p_drop=p_drop,
            add_layer_norm=add_layer_norm,
            activation=nn_activation,
            **kwargs)

        nn_ = MLP(
            dims=[dims[1]] + [dims[0]**2],
            p_drop=p_drop,
            add_layer_norm=add_layer_norm,
            activation=nn_activation,
            **kwargs)

        self.conv = NNConv(
            in_channels=dims[0],
            out_channels=dims[0],
            nn=nn_,
            aggr='add')

        self.drop = nn.Dropout(p_drop)

    def forward(self, x, edge_index, edge_attr):
        """Forward pass of the layer."""
        e = torch.cat([x[edge_index[0]], x[edge_index[1]], edge_attr], dim=-1)
        e = self.edge_attr_mlp(e)

        x = self.conv(x, edge_index, e)
        x = self.drop(x)
        return x, e


class NNConvNetwork(torch.nn.Module):
    """NNConv Network model."""
    def __init__(self, input_dims, output_dims, hidden_dims, layers=2, p_drop=0.0, activation='ReLU', **kwargs):
        super().__init__()

        self.convs = ModuleList()
        self.convs.append(
            NNConvLayer(
                input_dims=input_dims,
                output_dims=hidden_dims,
                hidden_dims=hidden_dims,
                p_drop=p_drop,
                nn_activation=activation,
                **kwargs))

        for _ in range(layers - 1):
            self.convs.append(
                NNConvLayer(
                    input_dims=hidden_dims,
                    output_dims=hidden_dims,
                    hidden_dims=hidden_dims,
                    p_drop=p_drop,
                    nn_activation=activation,
                    **kwargs))

        self.final_conv = MLP(
            dims=[hidden_dims[0], hidden_dims[0], output_dims[0]],
            activation=activation,
            **kwargs
        )

        self.activation = ActivationFunction(activation, **kwargs)

    def forward(self, data):
        """Forward pass of the model."""
        x, edge_index, e = data.x, data.edge_index, data.edge_attr

        for conv in self.convs:
            x, e = conv(x, edge_index, e)
            x = self.activation(x)
            e = self.activation(e)

        x = self.final_conv(x)

        return x


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
