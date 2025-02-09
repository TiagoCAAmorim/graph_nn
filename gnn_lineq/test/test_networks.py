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
        """Test mlp network block."""
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
        """Test edge mlp network block."""
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



    # def test_graph_conv_network(self):
    #     """Test GraphConvNetwork."""

    #     sample = self.samples.get_graph()
    #     model = GraphConvNetwork(
    #         input_dim=sample.num_node_features,
    #         output_dim=sample.num_features,
    #         hidden_dim=16,
    #         layers=5)
    #     result = model(sample)

    #     self.assertEqual(result.shape, sample.y.shape)
    #     self.assertFalse(sample.y.requires_grad)
    #     self.assertTrue(result.requires_grad)

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


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestNetworks('test_activation'))
    suite.addTest(TestNetworks('test_mlp'))
    # suite.addTest(TestNetworks('test_graph_conv_network'))

    runner = unittest.TextTestRunner()
    runner.run(suite)
