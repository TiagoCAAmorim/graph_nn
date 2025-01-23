"""
Test PyG installation.

Based on PyG's official tutorial:
    https://github.com/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial1/Tutorial1.ipynb
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid # type: ignore
from torch_geometric.nn import SAGEConv # type: ignore


def print_dataset(dataset):
    """Print dataset information."""
    print('=================================')
    print(dataset)
    print("number of graphs:\t\t",len(dataset))
    print("number of classes:\t\t",dataset.num_classes)
    print("number of node features:\t",dataset.num_node_features)
    print("number of edge features:\t",dataset.num_edge_features)
    print('=================================\n')

    print('=================================')
    print(dataset.data)
    print('=================================\n')

    print('=================================')
    print("edge_index:\t\t",dataset.data.edge_index.shape)
    print(dataset.data.edge_index)
    print("\n")
    print("train_mask:\t\t",dataset.data.train_mask.shape)
    print(dataset.data.train_mask)
    print(f'  Elements: {dataset.data.train_mask.sum()}')
    print("\n")
    print("val_mask:\t\t",dataset.data.val_mask.shape)
    print(dataset.data.val_mask)
    print(f'  Elements: {dataset.data.val_mask.sum()}')
    print("\n")
    print("test_mask:\t\t",dataset.data.test_mask.shape)
    print(dataset.data.test_mask)
    print(f'  Elements: {dataset.data.test_mask.sum()}')
    print("\n")
    print("x:\t\t",dataset.data.x.shape)
    print(dataset.data.x)
    print("\n")
    print("y:\t\t",dataset.data.y.shape)
    print(dataset.data.y)
    print('=================================\n')


class Net(torch.nn.Module):
    """Simple GNN model."""
    def __init__(self, dataset):
        super().__init__()

        self.conv_max = SAGEConv(dataset.num_features,
                             dataset.num_classes,
                             aggr="max")
        self.conv_mean = SAGEConv(dataset.num_features,
                             dataset.num_classes,
                             aggr="mean")

    def forward(self, x, edge_index):
        """Simple forward pass."""
        x_max = self.conv_max(x, edge_index)
        x_mean = self.conv_mean(x, edge_index)
        x = x_max + x_mean
        return F.log_softmax(x, dim=1)


def initialize_model(dataset):
    """Initialize a simple GNN model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    data = dataset.shuffle()[0]
    model, data = Net(dataset).to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    return model, data, optimizer


def run(model, data, optimizer):
    """Run the model."""

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        # Inverted train and test masks to use more samples in training
        loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask])
        loss.backward()
        optimizer.step()

    def test():
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        acc = []
        for _, mask in data('test_mask', 'val_mask', 'train_mask'):
            test_correct = pred[mask].eq(data.y[mask]).sum().item()
            acc.append(test_correct / mask.sum().item())
        return acc


    best_val_acc = 0
    for epoch in range(1,100):
        train()
        acc = test()
        train_acc = acc[0]
        val_acc = acc[1]
        test_acc = acc[2]
        best_val_acc = max(val_acc, best_val_acc)

        if (epoch % 10 == 0) or (best_val_acc == val_acc):
            log = f'Epoch: {epoch:03d}, '
            log += f'Train: {train_acc:.4f}, '
            log += f'Val: {val_acc:.4f}, '
            log += f'Test: {test_acc:.4f}, '
            log += f'Best: {best_val_acc:.4f} '
            if best_val_acc == val_acc:
                log += '(Best)'
            print(log)


def count_model_weights(model):
    """Count the number of weights in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """Main test function."""
    dataset = Planetoid(root="pyg/no_sync",name= "Cora")
    print_dataset(dataset)

    print('=================================')
    model, data, optimizer = initialize_model(dataset)
    print(model)
    print('=================================\n')

    weight_count = count_model_weights(model)
    print(f'Total number of weights in the model: {weight_count:,}')

    run(model, data, optimizer)
    print('Done!')

if __name__ == '__main__':
    main()
