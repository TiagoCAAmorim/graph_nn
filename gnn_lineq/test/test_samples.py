# pylint: disable=import-error, wrong-import-position
"""Test samples module."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))

import unittest
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from samples import LinEqSample, DynamicGraphDataset
from utils import plot_samples_error


class TestSamples(unittest.TestCase):
    """Test metrics module."""

    @classmethod
    def setUpClass(cls):
        cls.options = {
            'rank': 20,
            'diagonals': 7,
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
        cls.folder = Path(__file__).parent / '_plots'
        cls.folder.mkdir(exist_ok=True, parents=True)

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


    def test_noisy_sample(self):
        """Test sample generation."""
        _, axes = plt.subplots(2, 4, figsize=(16,8))
        axes = axes.flatten()

        noise_options = self.options.copy()
        mse = []
        std = [10**(i/4-2) for i in range(7)]
        for i in range(7):
            noise_options['std'] = std[i]

            last_mse = 0
            new_sample = LinEqSample(**noise_options)
            x_noise = []
            x_solve = []
            for _ in range(1000):
                matrix, x_true, b = new_sample.get(max_error=1E-6, max_iter=1000, throw_error=True)
                x_noise.append(x_true)
                x_solve.append(LinEqSample.solve(matrix, b))

            mse.append(F.mse_loss(
                torch.tensor(np.array(x_noise)),
                torch.tensor(np.array(x_solve))
            ))
            msg = f'Std: {std[i]:.1E}, MSE: {mse[i]:.1E}'
            self.assertGreater(mse[i], last_mse)
            last_mse = mse[i]

            plot_samples_error(None, (x_solve[:4], x_noise[:4]), ax=axes[i], title=msg)

        axes[-1].plot(std, mse, 'o-')
        axes[-1].set_xscale('log')
        axes[-1].set_yscale('log')
        axes[-1].set_xlabel('Noise Standard Deviation')
        axes[-1].set_ylabel('MSE')
        axes[-1].grid(which='both', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(self.folder / 'noisy_samples.png')


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestSamples('test_sample'))
    suite.addTest(TestSamples('test_big_sample'))
    suite.addTest(TestSamples('test_dataset'))
    suite.addTest(TestSamples('test_noisy_sample'))

    runner = unittest.TextTestRunner()
    runner.run(suite)
