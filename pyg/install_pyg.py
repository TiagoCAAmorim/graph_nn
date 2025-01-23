"""
Install PyG with the correct version of PyTorch.

Source: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
"""
import os
import subprocess
import torch

def main():
    """Install PyG"""
    os.environ['TORCH'] = torch.__version__
    print(f'Current torch verion: {torch.__version__}')

    subprocess.check_call([
        'pip',
        'install',
        'torch_geometric'])

    subprocess.check_call([
        'pip',
        'install',
        'pyg_lib',
        'torch_scatter',
        'torch_sparse',
        'torch_cluster',
        'torch_spline_conv',
        '-f',
        f'https://data.pyg.org/whl/torch-{torch.__version__}.html'])

if __name__ == '__main__':
    main()
