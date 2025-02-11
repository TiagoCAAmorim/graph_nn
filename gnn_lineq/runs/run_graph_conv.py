# pylint: disable=import-error, wrong-import-position
"""Train network."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))

import time
import random
import itertools

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

from networks import TransformerConvLayer, EncDecNetwork
from samples import LinEqSample, DynamicGraphDataset
from utils import plot_samples_error, plot_dataset_error, train_model, evaluate_model, plot_losses


def build_dataset(std=0.0):
    """Define dataset."""
    options = {
        'rank': 20,
        'diagonals': 7,
        'off_diagonal_abs_mean': 0.5,
        'symmetric': False,
        'width_range': (0.1,10.),
        'std': std
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


def build_model(params):
    """Build the model from params dict."""
    dataset = build_dataset()
    if 'pretrained_model' not in params:
        params['pretrained_model'] = None
    model = EncDecNetwork(
        input_dims=(dataset.num_node_features, dataset.num_edge_features),
        lat_dims=(params['node_lat_dim'], params['edge_lat_dim']),
        output_dim=1,
        process_block=TransformerConvLayer,
        n_process_blocks=params['n_process_blocks'],
        add_skip=params['add_skip'],
        mlp_layers=params['mlp_layers'],
        p_drop=params['p_drop'],
        add_layer_norm=params['add_layer_norm'],
        activation=params['activation'],
        heads=params['heads'],
        beta=params['beta'],
        pretrained_model=params['pretrained_model'],
    )
    return model


def train_and_evaluate(params, epochs=5, writer=None):
    """Train and evaluate the model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = build_dataset(std=params['std'])
    dataset_eval = build_dataset(std=0.0)
    loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=False)
    loader_eval = DataLoader(dataset_eval, batch_size=params['batch_size'], shuffle=False)

    model = build_model(params)

    model = model.to(device)
    if params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay'])
    else:
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay'])

    loss, best_model = train_model(
        model,
        loader,
        epochs=epochs,
        print_epochs=10,
        optimizer=optimizer,
        writer=writer)
    val_loss = evaluate_model(best_model, loader_eval)

    return loss, val_loss, best_model


def generate_combinations(param_grid, shuffle=True):
    """
    Generate all possible combinations of the values in the lists within a dictionary.

    Parameters:
    -----------
    - param_grid (dict): Dictionary of lists.
    - shuffle (bool): Shuffle the combinations.

    Returns:
    --------
    - list of dict: List of dictionaries with all possible combinations of the values.
    """
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    if shuffle:
        random.shuffle(combinations)
    return combinations


def random_search(param_grid, n_iter=10, epochs=5, writer_folder=None):
    """Random search for hyperparameter optimization."""
    best_params = None
    best_score = float('inf')

    params_list = generate_combinations(param_grid, shuffle=True)
    n_iter = min(n_iter, len(params_list))

    for i,params in enumerate(params_list[:n_iter]):
        time0 = time.time()

        writer = None
        if writer_folder is not None:
            writer = SummaryWriter(log_dir=writer_folder / f'iter_{i+1}')
        loss, score, model = train_and_evaluate(params, epochs=epochs, writer=writer)
        if writer_folder is not None:
            add_hparams(writer, params, score)
            writer.close()
            save_model_and_params(model, params, writer_folder / 'save' / f'model_{i+1}.pth')
            plot_results(model, loss, writer_folder / 'plots' / f'iter_{i+1}', writer)

        delta_time = time.time() - time0
        print(f'Iteration: {i+1}/{n_iter}, Score: {score:0.4g} ({delta_time:0.2f}s)')
        if score < best_score:
            best_score = score
            best_params = params

    return best_params, best_score


def add_hparams(writer, params, score):
    """Add hyperparameters to tensorboard."""
    hparams = {k: str(v) for k, v in params.items()}
    hparams['score'] = score
    writer.add_hparams(hparams, {})


def save_model_and_params(model, params, path):
    """Save the model's state dictionary and hyperparameters."""
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    state = {
        'model_state_dict': model.state_dict(),
        'params': params
    }
    torch.save(state, path)


def load_model_and_params(path):
    """Load the model's state dictionary and hyperparameters."""
    state = torch.load(path, weights_only=False)
    model = build_model(state['params'])
    model.load_state_dict(state['model_state_dict'])
    return model, state['params']


def plot_results(model, loss, folder, writer=None):
    """Plot the results."""
    dataset = build_dataset()
    folder.mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(1, 3, figsize=(16,8))
    axes = axes.flatten()

    plot_losses(loss, axes[0], title='MSE Loss')

    plot_dataset_error(model, dataset, axes[1], title='Error Histogram')

    samples = [dataset[i] for i in range(5)]
    plot_samples_error(model, samples, axes[2], title='Samples')

    plt.suptitle('GraphConvNetwork Results')
    plt.savefig(folder / 'Sample.png')
    plt.close()

    # Add the figure to TensorBoard
    if writer is not None:
        writer.add_figure('Results', fig)


def hparam_optimization(folder, epochs=100):
    """Search for best hyperparams."""
    param_grid = {
        'batch_size': [64], #[32, 64, 128, 256, 512],
        'optimizer': ['adam'], #['adam', 'rmsprop'],

        'lr': [1e-2, 1e-3], #[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
        'weight_decay': [1e-7], #[1e-4, 1e-5, 1e-6, 1e-7],

        'node_lat_dim':[16], #[8, 16, 32, 64],
        'edge_lat_dim':[16, 32], #[4, 8, 16, 32],
        'n_process_blocks': [2,3,4], #[1, 2, 4],
        'add_skip': [True], #[False, True],
        'mlp_layers': [4, 8], #[2, 4, 8],
        'p_drop': [0.0, 0.05], #[0.0, 0.1, 0.2],
        'add_layer_norm': [False], #[False, True],
        'activation': ['leakyrelu'], #['relu', 'tanh', 'sigmoid', 'leakyrelu'],
        'heads': [4], #[1, 2, 4, 8],
        'beta': [True], #[False, True],
        'std': [0.0, 1E-2, 2E-2, 3E-2]
    }

    # Test very big model
    # _, _, _ = train_and_evaluate({k:v[-1] for k,v in param_grid.items()}, epochs=10, writer=None)

    best_params, best_score = random_search(
        param_grid,
        n_iter=500,
        epochs=epochs,
        writer_folder=folder)
    print(f'Best Hyperparameters: {best_params}')
    print(f'Best Score: {best_score:0.4g}')

    return best_params

def run(params, name, folder, epochs=100):
    """Train the network."""

    writer = SummaryWriter(log_dir=folder / name)
    loss, best_score, model = train_and_evaluate(params, epochs=epochs, writer=writer)

    add_hparams(writer, params, best_score)
    save_model_and_params(model, params, folder / 'save' / f'{name}.pth')
    folder = Path(__file__).resolve().parent / '_plots'
    plot_results(model, loss, folder, writer)
    writer.close()

def run_best(folder):
    """Run the best model."""

    best_params = {
        'batch_size': 4,
        'optimizer': 'adam',
        'lr': 0.001,
        'weight_decay': 1E-7,
        'node_lat_dim': 16,
        'edge_lat_dim': 32,
        'n_process_blocks': 2,
        'add_skip': True,
        'mlp_layers': 6,
        'p_drop': 0.0,
        'add_layer_norm': False,
        'activation': 'leakyrelu',
        'heads': 2,
        'beta': True,
        'std': 0.0
    }

    run(best_params, 'best_5', folder, epochs=1000)

def main():
    """Run the main function."""
    experiment_name = 'TransformerConvNoise'
    folder = Path(__file__).resolve().parent / '_runs' / experiment_name
    folder.mkdir(exist_ok=True, parents=True)

    best_params = hparam_optimization(folder)
    print(best_params)
    # run(best_params, 'best_5', folder, epochs=1000)

    # run_best(folder)

    # model_name = 'best_2'
    # pre_model, pre_params = load_model_and_params(folder / 'save' / f'{model_name}.pth')
    # pre_params['pretrained_model'] = pre_model
    # pre_params['n_process_blocks'] = 6

    # run(pre_params, 'best_2B', folder, epochs=1000)

    # model_name = 'best_2B'
    # pre_model, pre_params = load_model_and_params(folder / 'save' / f'{model_name}.pth')
    # pre_params['pretrained_model'] = pre_model
    # pre_params['n_process_blocks'] = 12

    # run(pre_params, 'best_2B1', folder, epochs=1000)


if __name__ == '__main__':
    main()
