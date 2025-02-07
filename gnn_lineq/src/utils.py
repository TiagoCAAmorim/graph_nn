"""Assorted helper functions."""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_losses(losses, x_label='Epoch', y_label='MSE Loss', title=None):
    """Plot the loss curve."""
    plt.plot(losses['x'], losses['y'])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.yscale('log')
    plt.grid(which='both', linewidth=0.5)
    if title is not None:
        plt.title(title)


def plot_sample_error(model, sample, limits=(-1.5,1.5), sorted_values=True, title=None):
    """Plot the result of the model on a sample."""
    model.eval()

    device = next(model.parameters()).device
    with torch.no_grad():
        result = model(sample.to(device))

    real = sample.y.cpu().flatten().numpy()
    estimate = result.cpu().flatten().numpy()

    if sorted_values:
        sorted_indices = np.argsort(real)
    else:
        sorted_indices = np.arange(len(real))

    plt.plot(real[sorted_indices], label='Real')
    plt.plot(estimate[sorted_indices], label='Estimate')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(which='both', linewidth=0.5)
    plt.ylim(limits[0], limits[1])
    if title is None:
        residual = np.sqrt(np.mean(np.square(real - estimate)))
        title = f'RMS error: {residual:.4g}'
    plt.title(title)


def plot_dataset_error(model, dataset, title=None):
    """Plot the histogram of the errors on a dataset."""
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
    plt.hist(errors, bins=50)
    plt.xlabel('Relative Error')
    plt.ylabel('Count')
    plt.grid(which='both', linewidth=0.5)
    if title is None:
        residual = np.sqrt(np.mean(np.square(errors)))
        title = f'RMS error: {residual:.4g}'
    plt.title(title)


def train_model(model, loader, epochs=200, print_epoch_step=10, optimizer=None):
    """Train a model on a DataLoader."""
    print(f'Number of parameters: {count_parameters(model):,}')
    device = next(model.parameters()).device

    model = model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    losses = {'x':[], 'y':[]}
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
        losses['x'].append(epoch)
        losses['y'].append(total_loss / len(loader))
        if print_epoch_step > 0:
            if epoch % print_epoch_step == 0:
                print(f'Epoch: {epoch:,}, Loss: {loss.item():0.4g}')
    return losses
