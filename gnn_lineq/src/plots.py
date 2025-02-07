"""
Helper functions to generate plots.

To open html from WSL:
- Install wslu: sudo apt-get install wslu
- Add variable to ~/.bashrc: export BROWSER=wslview
"""

from torch_geometric.utils.convert import to_networkx

import matplotlib.pyplot as plt
import networkx as nx

from bokeh.models import MultiLine, HoverTool
from bokeh.palettes import viridis
from bokeh.plotting import figure, from_networkx, show, output_file, save

if __name__ == '__main__':
    import samples # pylint: disable=import-error


def plot_histograms(arrays, bins=30, figsize=(6, 1), max_height=20, title=None, file_path=None):
    """
    Plot histograms for a list of numpy arrays in a single column.

    Arguments:
    ----------
    - arrays (list of np.array): List of numpy arrays to plot.
    - bins (int): Number of bins for the histograms.
    - figsize (tuple): Size of the figure. The height is multiplied
        by the number of arrays to plot.
    - max_height (int): Maximum height of the figure. Default is 20.
    - title (str): Title of the plot. If None, no title is shown.
        Default is None.
    - file_path (str): Path to save the plot. If None, the plot is not
        saved. Default is None.

    Returns:
    --------
    - axes (list of plt.Axes): List of axes objects for the histograms.
    """
    num_arrays = len(arrays)
    figsize = (figsize[0], min(figsize[1]*(1+num_arrays), max_height))
    _, axes = plt.subplots(num_arrays, 1, figsize=figsize, sharex=True)

    if num_arrays == 1:
        axes = [axes]

    for i, data_ in enumerate(arrays):
        axes[i].hist(data_, bins=bins, edgecolor='black')
        axes[i].set_ylabel('Freq.')

    if title is not None:
        axes[0].set_title(title)

    axes[-1].set_xlabel('Value')
    plt.tight_layout()

    if file_path is not None:
        plt.savefig(file_path)

    return axes


def plot_graph(graph, ax=None, fig_size=(4,4), title=None, file_path=None):
    """
    Static plot of the given graph with Matplotlib.

    Arguments:
    ----------
    - graph (torch_geometric.data.Data): Graph to plot.
    - ax (plt.Axes): Axes object to plot the graph. If None, a new
        figure is created. Default is None.
    - fig_size (tuple): Size of the figure. Default is (4,4).
    - title (str): Title of the plot. If None, no title is shown.
        Default is None.
    - file_path (str): Path to save the plot. If None, the plot is not
        saved. Default is None.

    Returns:
    --------
    - ax (plt.Axes): Axes object with the plot.
    """
    vis = to_networkx(graph)

    node_index = list(range(graph.num_nodes))

    if ax is None:
        _, ax = plt.subplots(figsize=fig_size)

    nx.draw(
        vis,
        ax=ax,
        cmap=plt.get_cmap('Set3'),
        node_color=node_index,
        node_size=70,
        linewidths=6
    )

    if title is not None:
        ax.set_title(title)

    if file_path is not None:
        plt.savefig(file_path)

    return ax


def plot_graph_bokeh(graph, width=400, height=400, layout=None, title=None, file_path=None):
    """
    Interactive plot of the given graph with Bokeh.

    Source: https://docs.bokeh.org/en/latest/docs/user_guide/topics/graph.html
    """
    p = figure(
        width=width, height=height,
        x_range=(-1.2, 1.2), y_range=(-1.2, 1.2),
        x_axis_location=None, y_axis_location=None,
        tools="hover,pan,wheel_zoom,box_zoom,save,reset,help",
        tooltips="")
    p.grid.grid_line_color = None

    hover = HoverTool()
    hover.tooltips = [("index", "$index")]
    for k in graph.node_attrs():
        hover.tooltips.append((k, f'@{k}'+'{0.4f}'))
    p.add_tools(hover)

    graph_x = to_networkx(graph, node_attrs=['x','y'], edge_attrs=['edge_attr'])
    if layout is None:
        layout = nx.spring_layout
    graph_viz = from_networkx(graph_x, layout, scale=1.0, center=(0,0))
    p.renderers.append(graph_viz) # pylint: disable=no-member

    graph_viz.node_renderer.data_source.data['index'] = list(range(graph.num_nodes))
    graph_viz.node_renderer.data_source.data['colors'] =  viridis(graph.num_nodes) # pylint: disable=no-member
    graph_viz.edge_renderer.glyph = MultiLine(line_color="gray", line_alpha=0.8, line_width=3)
    graph_viz.node_renderer.glyph.update(size=15, fill_color="colors")

    if title is not None:
        p.title.text = title

    if file_path is not None:
        output_file(file_path)
        save(p)
    else:
        show(p)



def test_multi_histogram_plot():
    """Test plot with multiple histograms."""
    data_ = []
    for _ in range(6):
        data_.append(
            samples.LinEqSample.number_generator(1000, width=(0.1,10.)) # pylint: disable=possibly-used-before-assignment
        )
    _,_ = plot_histograms(data_, bins=30, title='Samples Histograms')
    plt.show()


def test_simple_graph_plot():
    """Test static graph plot."""
    dataset = samples.LinEqSample(
        rank=5,
        diagonals=2,
        off_diagonal_abs_mean=0.5,
        symmetric=False,
        width_range=(0.1,10.)
    )
    graph = dataset.get_graph()
    print('Edge index:')
    print(graph.edge_index)
    plot_graph(graph, fig_size=(4,4))
    plt.show()


def test_simple_graph_plot_panel():
    """Test building panel of graph plots."""
    _, ax = plt.subplots(2, 4, figsize=(12, 6))
    ax = ax.flatten()
    for i in range(8):
        dataset = samples.LinEqSample(
            rank=5,
            diagonals=i+2,
            off_diagonal_abs_mean=0.5,
            symmetric=False,
            width_range=(0.1,10.)
        )
        graph = dataset.get_graph()
        plot_graph(graph, ax=ax[i], title=f'Diagonals: {i+2}')
    plt.show()


def test_bokeh_graph_plot():
    """Test bokeh graph plot."""
    dataset = samples.LinEqSample(
        rank=5,
        diagonals=4,
        off_diagonal_abs_mean=0.5,
        symmetric=False,
        width_range=(0.1,10.)
    )
    graph = dataset.get_graph()
    print('Edge index:')
    print(graph.edge_index)
    print('X:')
    print(graph.x)
    print('Y:')
    print(graph.y)
    plot_graph_bokeh(graph, title='Graph Visualization')
    plt.show()


if __name__ == '__main__':
    test_multi_histogram_plot()
    test_simple_graph_plot()
    test_simple_graph_plot_panel()
    test_bokeh_graph_plot()
