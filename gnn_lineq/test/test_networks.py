# pylint: disable=import-error, wrong-import-position
"""Test networks module."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))

import random
import unittest
import torch
from torch_geometric.loader import DataLoader

from samples import LinEqSample, DynamicGraphDataset
from networks import ActivationFunction, MLP, EdgeMLP
from networks import NNConvLayer, TransformerConvLayer, EncDecNetwork
from utils import count_parameters


class TestNetworks(unittest.TestCase):
    """Test networks module."""

    @classmethod
    def setUpClass(cls):
        options = {
            'rank': 5,
            'diagonals': 4,
            'off_diagonal_abs_mean': 0.5,
            'symmetric': False,
            'width_range': (0.1,10.)
        }
        cls.samples = LinEqSample(**options)

        dataset = DynamicGraphDataset(
            dataset_len=100,
            sample_function=cls.samples.get_graph,
            max_error=1E-6,
            max_iter=1000,
            throw_error=True
        )
        cls.loader = DataLoader(dataset, batch_size=32, shuffle=False)


    def test_activation(self):
        """Test activation network."""
        sample = torch.randn(10, 5)
        model = ActivationFunction(None)
        result = model(sample)
        self.assertEqual(result.shape, sample.shape)
        for i in range(10):
            for j in range(5):
                self.assertEqual(result[i,j], sample[i,j])

        model = ActivationFunction('ReLu')
        result = model(sample)
        self.assertEqual(result.shape, sample.shape)
        self.assertEqual(count_parameters(model), 0)

        model = ActivationFunction('LeakyReLu', negative_slope=0.1)
        result = model(sample)
        self.assertEqual(result.shape, sample.shape)

        with self.assertRaises(ValueError):
            model = ActivationFunction('Unknown')
        with self.assertRaises(TypeError):
            model = ActivationFunction('LeakyReLu', sl=0.1)


    def test_mlp(self):
        """Test MLP block."""
        sample = torch.randn(10, 5)

        dims=[5, 10, 6]
        model = MLP(
            dims=dims,
            activation='ReLU',
        )
        result = model(sample)
        self.assertEqual(result.shape, (sample.shape[0], dims[-1]))
        n_weigths = sum((dims[i]+1)*dims[i+1] for i in range(len(dims)-1))
        self.assertEqual(count_parameters(model), n_weigths)

        model = MLP(
            dims=dims,
            activation='ReLU',
            add_layer_norm=True,
        )
        result = model(sample)
        self.assertEqual(result.shape, (sample.shape[0], dims[-1]))
        n_weigths += dims[-1]*2
        self.assertEqual(count_parameters(model), n_weigths)


    def test_edge_mlp(self):
        """Test EdgeMLP block."""
        sample = TestNetworks.get_random_sample(
            n_nodes=5,
            n_edges=10,
            node_features=3,
            edge_features=2
        )
        model = EdgeMLP(
            dims=[3, 2],
            output_dims=[4, 6],
            p_drop=0.0,
            add_layer_norm=True,
            activation='ReLU',
        )
        result = model(*sample)
        self.assertEqual(result.shape, (10, 6))
        dims = [2*3+2, 4, 6]
        n_weigths = sum((dims[i]+1)*dims[i+1] for i in range(len(dims)-1))
        n_weigths += dims[-1]*2
        self.assertEqual(count_parameters(model), n_weigths)


    def test_nn_conv_layer(self):
        """Test NNConvLayer block."""
        sample = TestNetworks.get_random_sample(
            n_nodes=5,
            n_edges=10,
            node_features=3,
            edge_features=2
        )
        model = NNConvLayer(
            dims=[sample[0].shape[1], sample[2].shape[1]],
            edge_mlp_layers=2,
            p_drop=0.0,
            add_layer_norm=True,
            activation='ReLU',
        )
        result = model(*sample)
        self.assertEqual(result[0].shape, sample[0].shape)
        self.assertEqual(result[1].shape, sample[2].shape)

        print(f'{count_parameters(model):,} parameters')
        print(model)
        TestNetworks.print_model(model)


    def test_transformer_conv_layer(self):
        """Test TransformerConvLayer block."""
        sample = TestNetworks.get_random_sample(
            n_nodes=5,
            n_edges=10,
            node_features=3,
            edge_features=2
        )
        model = TransformerConvLayer(
            dims=[sample[0].shape[1], sample[2].shape[1]],
            heads=3,
            beta=True,
            edge_mlp_layers=2,
            p_drop=0.0,
            add_layer_norm=True,
            activation='ReLU',
        )
        result = model(*sample)
        self.assertEqual(result[0].shape, sample[0].shape)
        self.assertEqual(result[1].shape, sample[2].shape)

        print(f'{count_parameters(model):,} parameters')
        print(model)
        TestNetworks.print_model(model)


    def test_nn_conv_network(self):
        """Test Encoder-Decorder network with NNConvLayer."""
        sample = TestNetworks.get_random_sample(
            n_nodes=50,
            n_edges=100,
            node_features=3,
            edge_features=5
        )

        latent_dims = [16, 8]
        output_dim = 4

        model = EncDecNetwork(
            input_dims=(sample[0].shape[1], sample[2].shape[1]),
            lat_dims=latent_dims,
            output_dim=output_dim,
            process_block=NNConvLayer,
            n_process_blocks=2,
            add_skip=True,
            mlp_layers=2,
            p_drop=0.0,
            add_layer_norm=False,
            activation='relu',
        )

        result = model(*sample)
        self.assertEqual(result.shape, (sample[0].shape[0],output_dim))

        print(f'{count_parameters(model):,} parameters')
        print(model)
        TestNetworks.print_model(model)


    def test_transformer_conv_network(self):
        """Test Encoder-Decorder network with TransformerConvLayer."""
        sample = TestNetworks.get_random_sample(
            n_nodes=50,
            n_edges=100,
            node_features=3,
            edge_features=5
        )

        latent_dims = [16, 8]
        output_dim = 4

        model = EncDecNetwork(
            input_dims=(sample[0].shape[1], sample[2].shape[1]),
            lat_dims=latent_dims,
            output_dim=output_dim,
            process_block=TransformerConvLayer,
            n_process_blocks=2,
            add_skip=True,
            mlp_layers=2,
            p_drop=0.0,
            add_layer_norm=False,
            activation='relu',
            heads=3,
            beta=True,
        )

        result = model(*sample)
        self.assertEqual(result.shape, (sample[0].shape[0],output_dim))

        print(f'{count_parameters(model):,} parameters')
        print(model)
        TestNetworks.print_model(model)


    @staticmethod
    def get_random_sample(n_nodes=10, n_edges=20, node_features=5, edge_features=6):
        """Get a random sample."""
        nodes = torch.randn(n_nodes, node_features)

        all_pairs = [(i, j) for i in range(n_nodes) for j in range(n_nodes)]
        if n_edges > len(all_pairs):
            raise ValueError(f'Maximum number of edges: {len(all_pairs)}.')
        edge_index = random.sample(all_pairs, n_edges)
        edge_index.sort()
        edge_index = torch.tensor(edge_index).t().to(torch.long)
        edge_attr = torch.randn(edge_index.shape[1], edge_features)
        return (nodes, edge_index, edge_attr)


    @staticmethod
    def print_model(model, level=0, max_level=1, master='0'):
        """Print model."""
        if hasattr(model, 'children') and level < max_level:
            for i, layer in enumerate(model.children()):
                TestNetworks.print_model(
                    layer,
                    level=level+1,
                    max_level=max_level,
                    master=master+'.'+str(i)
                )
        else:
            print()
            print(f'{master} - {count_parameters(model):,} parameters')
            print(model)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestNetworks('test_activation'))
    suite.addTest(TestNetworks('test_mlp'))
    suite.addTest(TestNetworks('test_edge_mlp'))
    suite.addTest(TestNetworks('test_nn_conv_layer'))
    suite.addTest(TestNetworks('test_transformer_conv_layer'))
    suite.addTest(TestNetworks('test_nn_conv_network'))
    suite.addTest(TestNetworks('test_transformer_conv_network'))

    runner = unittest.TextTestRunner()
    runner.run(suite)
