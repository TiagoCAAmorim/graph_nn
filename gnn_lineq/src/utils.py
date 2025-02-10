"""Assorted helper functions."""

from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_losses(losses, ax=None, fig_size=(4,4), title=None, file_path=None):
    """Plot the loss curve."""
    if ax is None:
        _, ax = plt.subplots(figsize=fig_size)

    ax.plot(losses['x'], losses['y'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_yscale('log')
    ax.grid(which='both', linewidth=0.5)
    if title is not None:
        ax.set_title(title)

    if file_path is not None:
        plt.tight_layout()
        plt.savefig(file_path)

    return ax


def plot_samples_error(model, samples, ax=None, fig_size=(4,4), title=None, file_path=None):
    """Plot the result of the model for various samples."""
    if ax is None:
        _, ax = plt.subplots(figsize=fig_size)

    model.eval()
    device = next(model.parameters()).device
    results = []
    with torch.no_grad():
        for sample in samples:
            results.append(model(sample.to(device)))

    labels = ['Real', 'Estimate']
    for sample, result in zip(samples, results):
        real = sample.y.cpu().flatten().numpy()
        estimate = result.cpu().flatten().numpy()
        sorted_indices = np.argsort(real)

        base_line, = ax.plot(real[sorted_indices], label=labels[0])
        last_color = base_line.get_color()
        ax.plot(estimate[sorted_indices], label=labels[1], linestyle='--', color=last_color)
        labels = [None, None]

    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(which='both', linewidth=0.5)
    if title is None:
        residual = np.sqrt(np.mean(np.square(real - estimate)))
        title = f'RMS error: {residual:.4g}'
    ax.set_title(title)

    if file_path is not None:
        plt.tight_layout()
        plt.savefig(file_path)

    return ax


def plot_dataset_error(model, dataset, ax=None, fig_size=(4,4), title=None, file_path=None):
    """Plot the histogram of the errors on a dataset."""
    if ax is None:
        _, ax = plt.subplots(figsize=fig_size)

    device = next(model.parameters()).device
    model.eval()

    real = []
    estimate = []
    with torch.no_grad():
        for sample in dataset:
            result = model(sample.to(device))
            real.append(sample.y.cpu().flatten().numpy())
            estimate.append(result.cpu().flatten().numpy())

    real = np.concatenate(real)
    estimate = np.concatenate(estimate)

    errors = real - estimate
    ax.hist(errors, bins=50)
    ax.set_xlabel('Relative Error')
    ax.set_ylabel('Count')
    ax.grid(which='both', linewidth=0.5)
    if title is None:
        residual = np.sqrt(np.mean(np.square(errors)))
        title = f'RMS error: {residual:.4g}'
    ax.set_title(title)

    if file_path is not None:
        plt.tight_layout()
        plt.savefig(file_path)

    return ax


def train_model(model, loader, epochs=200, print_epochs=10, optimizer=None, writer=None):
    """Train a model on a DataLoader."""
    print(f'Number of parameters: {count_parameters(model):,}')
    device = next(model.parameters()).device

    model = model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    print_epoch_step = max(1, epochs // print_epochs)

    losses = {'x':[], 'y':[]}
    best_loss = float('inf')
    best_state = None

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            data = batch.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.mse_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        losses['x'].append(epoch)
        losses['y'].append(avg_loss)
        if writer:
            writer.add_scalar('Loss/train', avg_loss, epoch)
        if print_epoch_step > 0:
            if epoch % print_epoch_step == 0:
                print(f'Epoch: {epoch:,}, Avg.Loss: {avg_loss:0.4g}')
        if (avg_loss < best_loss) or (best_state is None):
            print(f'   New best at epoch {epoch:,}, Avg.Loss: {avg_loss:0.4g}')
            best_loss = avg_loss
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    return losses, model

def evaluate_model(model, loader):
    """Evaluate a model on a DataLoader."""
    device = next(model.parameters()).device

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            data = batch.to(device)
            output = model(data)
            loss = F.mse_loss(output, data.y)
            total_loss += loss.item()
    return total_loss / len(loader)
