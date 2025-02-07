# pylint: disable=import-error, wrong-import-position
"""Test plots module."""

import sys
import os
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))

import unittest
import matplotlib.pyplot as plt

from samples import LinEqSample
from plots import plot_histograms, plot_graph, plot_graph_bokeh


class TestPlots(unittest.TestCase):
    """Test plots module."""

    @classmethod
    def setUpClass(cls):
        options = {
            'rank': 5,
            'diagonals': 4,
            'off_diagonal_abs_mean': 0.5,
            'symmetric': False,
            'width_range': (0.1,10.)
        }
        cls.dataset = LinEqSample(**options)
        cls.folder = Path(__file__).parent / '_plots'
        cls.folder.mkdir(exist_ok=True, parents=True)


    def test_multi_histogram_plot(self):
        """Test plot with multiple histograms."""
        file_path = self.folder / 'histograms.png'
        data_ = []
        for _ in range(6):
            data_.append(
                LinEqSample.number_generator(1000, width=(0.1,10.))
            )
        _ = plot_histograms(
            data_,
            bins=30,
            title='Samples Histograms',
            file_path=file_path
        )
        self.assertTrue(file_path.exists())
        self.assertTrue(self.is_file_new(file_path))


    def test_simple_graph_plot(self):
        """Test static graph plot."""
        file_path = self.folder / 'graph_static.png'
        graph = self.dataset.get_graph()
        print('\n=== Simple Plot Example ===')
        print('Edge index:')
        print(graph.edge_index)
        plot_graph(
            graph,
            fig_size=(4,4),
            file_path=file_path
        )
        plt.close()
        self.assertTrue(file_path.exists())
        self.assertTrue(self.is_file_new(file_path))


    def test_simple_graph_plot_panel(self):
        """Test building panel of graph plots."""
        file_path = self.folder / 'graph_panel.png'
        _, ax = plt.subplots(2, 4, figsize=(12, 6))
        ax = ax.flatten()
        for i in range(8):
            dataset = LinEqSample(
                rank=5,
                diagonals=i+2,
                off_diagonal_abs_mean=0.5,
                symmetric=False,
                width_range=(0.1,10.)
            )
            graph = dataset.get_graph()
            plot_graph(graph, ax=ax[i], title=f'Diagonals: {i+2}')
        plt.savefig(file_path)
        plt.close()
        self.assertTrue(file_path.exists())
        self.assertTrue(self.is_file_new(file_path))


    def test_bokeh_graph_plot(self):
        """Test bokeh graph plot."""
        file_path = self.folder / 'graph_bokeh.html'
        graph = self.dataset.get_graph()
        print('\n=== Bokeh Example ===')
        print('Edge index:')
        print(graph.edge_index)
        print('X:')
        print(graph.x)
        print('Y:')
        print(graph.y)
        plot_graph_bokeh(
            graph,
            title='Graph Visualization',
            file_path=file_path
        )
        self.assertTrue(file_path.exists())
        self.assertTrue(self.is_file_new(file_path))


    def is_file_new(self, file_path):
        """Check if a file is new (less than 1 minute old)."""
        current_time = time.time()
        file_mod_time = os.path.getmtime(file_path)
        return (current_time - file_mod_time) < 60


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestPlots('test_multi_histogram_plot'))
    suite.addTest(TestPlots('test_simple_graph_plot'))
    suite.addTest(TestPlots('test_simple_graph_plot_panel'))
    suite.addTest(TestPlots('test_bokeh_graph_plot'))

    runner = unittest.TextTestRunner()
    runner.run(suite)
