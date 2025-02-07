# pylint: disable=import-error, wrong-import-position
"""Test samples module."""

import unittest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))
from samples import LinEqSample, DynamicGraphDataset


class TestSamples(unittest.TestCase):
    """Test metrics module."""

    @classmethod
    def setUpClass(cls):
        cls.options = {
            'rank': 5,
            'diagonals': 4,
            'off_diagonal_abs_mean': 0.5,
            'symmetric': False,
            'width_range': (0.1,10.)
        }
        cls.options_big = {
            'rank': 200,
            'diagonals': 30,
            'off_diagonal_abs_mean': 0.5,
            'symmetric': False,
            'width_range': (0.1,10.)
        }

    def test_sample(self):
        """Test sample generation."""
        new_sample = LinEqSample(**self.options)
        matrix, x_true, b = new_sample.get(max_error=1E-6, max_iter=1000, throw_error=True)

        self.assertEqual(matrix.shape, (self.options['rank'], self.options['rank']))
        self.assertEqual(x_true.shape, (self.options['rank'],))
        self.assertEqual(b.shape, (self.options['rank'],))

        residual = LinEqSample.calculate_residual(matrix, x_true, b)
        self.assertEqual(residual.shape, (self.options['rank'],))

        residual = LinEqSample.calculate_residual(matrix, x_true, b, aggr='rms')
        self.assertLess(residual, 1E-6)


    def test_big_sample(self):
        """Test big sample generation."""
        new_sample = LinEqSample(**self.options_big)
        matrix, x_true, b = new_sample.get(max_error=1E-20, max_iter=100, throw_error=False)
        residual = LinEqSample.calculate_residual(matrix, x_true, b, aggr='rms')
        self.assertGreater(residual, 1E-20)

        with self.assertRaises(ValueError):
            matrix, x_true, b = new_sample.get(max_error=1E-20, max_iter=100, throw_error=True)


    def test_dataset(self):
        """Test creating a dataset."""
        samples = LinEqSample(**self.options)

        dataset = DynamicGraphDataset(
            dataset_len=100,
            sample_function=samples.get_graph,
            max_error=1E-6,
            max_iter=1000,
            throw_error=True
        )

        self.assertEqual(len(dataset), 100)
        self.assertEqual(dataset[0].x.shape, (self.options['rank'], 1))
        self.assertEqual(dataset[0].y.shape, (self.options['rank'], 1))

        n_edges = sum(self.options['rank']-(i+1)//2 for i in range(self.options['diagonals']))
        self.assertEqual(dataset[0].edge_index.shape, (2, n_edges))
        self.assertEqual(dataset[0].edge_attr.shape, (n_edges, 1))


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestSamples('test_sample'))
    suite.addTest(TestSamples('test_big_sample'))
    suite.addTest(TestSamples('test_dataset'))

    runner = unittest.TextTestRunner()
    runner.run(suite)
