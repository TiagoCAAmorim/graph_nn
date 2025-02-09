# pylint: disable=line-too-long, too-many-arguments, unused-import
"""Module with the definition of the GNN model."""

import torch
from torch import nn
from torch.nn import ModuleList
# import torch.nn.functional as F

from torch_geometric.nn import GCN, summary, Sequential
from torch_geometric.nn.conv import NNConv, TransformerConv, PDNConv


# MARK: Activation functions
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
            negative_slope = kwargs.get('negative_slope', 0.01)
            self.activation = nn.LeakyReLU(negative_slope)
        elif activation.lower() == 'elu':
            alpha = kwargs.get('alpha', 1.0)
            self.activation = nn.ELU(alpha)
        else:
            msg = f'Unknown activation function: {activation}.'
            msg +=" Use `None`, 'ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU' or 'ELU'."
            raise ValueError(msg)

    def forward(self, x):
        """Forward pass of the activation function."""
        return self.activation(x)


# MARK: MLP
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


# MARK: EdgeMLP
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

    def forward(self, x, edge_index=None, edge_attr=None):
        """Forward pass of the model."""
        if edge_index is None:
            x, edge_index, edge_attr = x.x, x.edge_index, x.edge_attr
        e = torch.cat([x[edge_index[0]], x[edge_index[1]], edge_attr], dim=-1)
        e = self.mlp(e)
        return e


# MARK: NNConvLayer
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
    - nn_activation (str): Activation function to use for the NNConv nn argument
        and EdgeMLP. Default is `None`.
    - **kwargs: Additional arguments for the activation function
    """
    def __init__(self, dims, edge_mlp_layers=2, p_drop=0.0, add_layer_norm=False, activation=None, **kwargs):
        super().__init__()

        self.edge_attr_mlp = EdgeMLP(
            dims=dims,
            output_dims=edge_mlp_layers*[dims[1]],
            p_drop=p_drop,
            add_layer_norm=add_layer_norm,
            activation=activation,
            **kwargs)

        nn_ = MLP(
            dims=[dims[1]] + [dims[0]**2],
            p_drop=p_drop,
            add_layer_norm=add_layer_norm,
            activation=activation,
            **kwargs)

        self.conv = NNConv(
            in_channels=dims[0],
            out_channels=dims[0],
            nn=nn_,
            aggr='add')

        self.drop = nn.Dropout(p_drop)

    def forward(self, x, edge_index=None, edge_attr=None):
        """Forward pass of the model."""
        if edge_index is None:
            x, edge_index, edge_attr = x.x, x.edge_index, x.edge_attr
        edge_attr = self.edge_attr_mlp(x, edge_index, edge_attr)
        x = self.conv(x, edge_index, edge_attr)
        x = self.drop(x)
        return x, edge_attr


# MARK: TransformerConvLayer
class TransformerConvLayer(torch.nn.Module):
    """
    Single TransformerConv layer + Edge Attribute MLP.

    Parameters:
    -----------
    - dims (tuple): Dimensions of the input and output (nodes, edges). Assumes
        output dimensions are equal to the input dimensions.
    - heads (int): Number of multi-head-attentions. Default is 1.
    - beta (bool): If set, will combine aggregation and skip information.
        Default is False.
    - edge_mlp_layers (int): Number of layers for the edge MLP. Default is 2.
    - p_drop (float): Dropout probability. Default is 0.0.
    - add_layer_norm (bool): If True, a layer normalization is added after
        the last layer. Default is False.
    - nn_activation (str): Activation function to use for the NNConv nn argument
        and EdgeMLP. Default is `None`.
    - **kwargs: Additional arguments for the activation function
    """
    def __init__(self, dims, heads=1, beta=False, edge_mlp_layers=2, p_drop=0.0, add_layer_norm=False, activation=None, **kwargs):
        super().__init__()

        self.edge_attr_mlp = EdgeMLP(
            dims=dims,
            output_dims=edge_mlp_layers*[dims[1]],
            p_drop=p_drop,
            add_layer_norm=add_layer_norm,
            activation=activation,
            **kwargs)

        self.conv = TransformerConv(
            in_channels=dims[0],
            out_channels=dims[0],
            heads=heads,
            concat=False,
            beta=beta,
            dropout=p_drop,
            edge_dim=dims[1])

    def forward(self, x, edge_index=None, edge_attr=None):
        """Forward pass of the model."""
        if edge_index is None:
            x, edge_index, edge_attr = x.x, x.edge_index, x.edge_attr
        edge_attr = self.edge_attr_mlp(x, edge_index, edge_attr)
        x = self.conv(x, edge_index, edge_attr)
        return x, edge_attr


# MARK: EncDecNetwork
class EncDecNetwork(torch.nn.Module):
    """
    Encoder Decoder Network model.

    Parameters:
    -----------
    - input_dims (tuple): Dimensions of the input (nodes, edges).
    - lat_dims (tuple): Dimensions of the latent space (nodes, edges).
    - output_dim (int): Dimension of the output (nodes).
    - process_block (torch.nn.Module): Process block to use.
    - n_process_blocks (int): Number of process blocks. Default is 2.
    - add_skip (bool): If True, a skip connection is added to the process block.
        Default is False.
    - mlp_layers (int): Number of layers for the MLP. Default is 2.
    - p_drop (float): Dropout probability. Default is 0.0.
    - add_layer_norm (bool): If True, a layer normalization is added after
        the last layer. Default is False.
    - activation (str): Activation function to use. Default is `None`.
    - **kwargs: Additional arguments for the process block and activation function.
    """
    def __init__(self, input_dims, lat_dims, output_dim, process_block, n_process_blocks=2, add_skip=False,
                 mlp_layers=2, p_drop=0.0, add_layer_norm=False, activation=None, **kwargs):
        super().__init__()

        self.enconder_nodes = MLP(
            dims=[input_dims[0]] + mlp_layers*[lat_dims[0]],
            p_drop=p_drop,
            add_layer_norm=add_layer_norm,
            activation=activation,
            final_activation=False,
            **kwargs
        )

        self.enconder_edges = MLP(
            dims=[input_dims[1]] + mlp_layers*[lat_dims[1]],
            p_drop=p_drop,
            add_layer_norm=add_layer_norm,
            activation=activation,
            final_activation=False,
            **kwargs
        )

        self.process_block = ModuleList()
        for _ in range(n_process_blocks):
            self.process_block.append(
                process_block(
                    dims=lat_dims,
                    edge_mlp_layers=mlp_layers,
                    p_drop=p_drop,
                    add_layer_norm=add_layer_norm,
                    activation=activation,
                    **kwargs)
            )

        self.decoder_nodes = MLP(
            dims=mlp_layers*[lat_dims[0]] + [output_dim],
            p_drop=p_drop,
            add_layer_norm=add_layer_norm,
            activation=activation,
            final_activation=False,
            **kwargs
        )

        self.add_skip = add_skip


    def forward(self, x, edge_index=None, edge_attr=None):
        """Forward pass of the model."""
        if edge_index is None:
            x, edge_index, edge_attr = x.x, x.edge_index, x.edge_attr
        x = self.enconder_nodes(x)
        e = self.enconder_edges(edge_attr)

        for layer in self.process_block:
            x_, e = layer(x, edge_index, e)
            if self.add_skip:
                x = x + x_
            else:
                x = x_

        x = self.decoder_nodes(x)

        return x


if __name__ == '__main__':
    print(__doc__)
