# pylint: disable=import-error, wrong-import-position
"""Test plots module."""

import unittest
from pathlib import Path
import sys

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))
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


    def test_multi_histogram_plot(self):
        """Test plot with multiple histograms."""
        data_ = []
        for _ in range(6):
            data_.append(
                LinEqSample.number_generator(1000, width=(0.1,10.))
            )
        _ = plot_histograms(data_, bins=30, title='Samples Histograms')
        plt.show()


    def test_simple_graph_plot(self):
        """Test static graph plot."""
        graph = self.dataset.get_graph()
        print('Edge index:')
        print(graph.edge_index)
        plot_graph(graph, fig_size=(4,4))
        plt.show()


    def test_simple_graph_plot_panel(self):
        """Test building panel of graph plots."""
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
        plt.show()


    def test_bokeh_graph_plot(self):
        """Test bokeh graph plot."""
        graph = self.dataset.get_graph()
        print('Edge index:')
        print(graph.edge_index)
        print('X:')
        print(graph.x)
        print('Y:')
        print(graph.y)
        plot_graph_bokeh(graph, title='Graph Visualization')
        plt.show()


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestPlots('test_multi_histogram_plot'))
    suite.addTest(TestPlots('test_simple_graph_plot'))
    suite.addTest(TestPlots('test_simple_graph_plot_panel'))
    suite.addTest(TestPlots('test_bokeh_graph_plot'))

    runner = unittest.TextTestRunner()
    runner.run(suite)
