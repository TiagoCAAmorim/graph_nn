# pylint: disable=import-error, wrong-import-position
"""Test networks module."""

import unittest
from pathlib import Path
import sys

from torch_geometric.loader import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))
from samples import LinEqSample, DynamicGraphDataset
from networks import GraphConvNetwork


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


    def test_graph_conv_network(self):
        """Test GraphConvNetwork."""

        sample = self.samples.get_graph()
        model = GraphConvNetwork(
            input_dim=sample.num_node_features,
            output_dim=sample.num_features,
            hidden_dim=16,
            layers=5)
        result = model(sample)

        self.assertEqual(result.shape, sample.y.shape)
        self.assertFalse(sample.y.requires_grad)
        self.assertTrue(result.requires_grad)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestNetworks('test_graph_conv_network'))

    runner = unittest.TextTestRunner()
    runner.run(suite)
