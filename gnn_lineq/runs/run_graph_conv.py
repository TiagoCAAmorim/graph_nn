# pylint: disable=import-error, wrong-import-position
"""Train network."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))

import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

from networks import TransformerConvLayer, EncDecNetwork
from samples import LinEqSample, DynamicGraphDataset
from utils import plot_samples_error, plot_dataset_error, train_model, plot_losses


def build_dataset():
    """Define dataset."""
    options = {
        'rank': 50,
        'diagonals': 10,
        'off_diagonal_abs_mean': 0.5,
        'symmetric': False,
        'width_range': (0.1,10.)
    }
    samples = LinEqSample(**options)

    dataset = DynamicGraphDataset(
        dataset_len=1000,
        sample_function=samples.get_graph,
        max_error=1E-6,
        max_iter=1000,
        throw_error=False
    )

    return dataset


def main():
    """Train the network."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    dataset = build_dataset()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    model = EncDecNetwork(
        input_dims=(dataset.num_node_features, dataset.num_edge_features),
        lat_dims=(8, 4),
        output_dim=1,
        process_block=TransformerConvLayer,
        n_process_blocks=2,
        add_skip=True,
        mlp_layers=3,
        p_drop=0.00,
        add_layer_norm=False,
        activation='relu',
        # negative_slope=0.1,
        heads=3,
        beta=False,
    )

    print('\nTraining the model...')
    model = model.to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=5e-4)
    loss = train_model(model, loader, epochs=200, print_epochs=20, optimizer=optimizer)

    print('Plotting the results...')
    folder = Path(__file__).resolve().parent / '_plots'
    folder.mkdir(exist_ok=True, parents=True)

    _,axes = plt.subplots(1, 3, figsize=(16,8))
    axes = axes.flatten()

    plot_losses(loss, axes[0], title='RMSE Loss')

    plot_dataset_error(model, dataset, axes[1], title='Error Histogram')

    samples = [dataset[i] for i in range(5)]
    plot_samples_error(model, samples, axes[2], title='Samples')

    plt.suptitle('GraphConvNetwork Results')
    plt.savefig(folder / 'Sample.png')
    plt.close()



if __name__ == '__main__':
    main()
