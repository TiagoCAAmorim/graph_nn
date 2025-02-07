"""Train network."""

from pathlib import Path
import sys

import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))
from networks import GraphConvNetwork
from samples import LinEqSample, DynamicGraphDataset
from utils import plot_sample_error, plot_dataset_error, train_model, plot_losses


def build_dataset():
    """Define dataset."""
    options = {
        'rank': 5,
        'diagonals': 4,
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
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    print('Training the model...')
    model = GraphConvNetwork(
            input_dim=dataset.num_node_features,
            output_dim=dataset.num_features,
            hidden_dim=16,
            layers=5)
    model = model.to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss = train_model(model, loader, epochs=200, print_epoch_step=10, optimizer=optimizer)

    print('Plotting the results...')
    plot_losses(loss)
    plt.savefig('Loss.png')
    plt.close()

    plot_sample_error(model, dataset[0])
    plt.savefig('Sample.png')
    plt.close()

    plot_dataset_error(model, dataset)
    plt.savefig('ErrorHistogram.png')
    plt.close()




if __name__ == '__main__':
    main()