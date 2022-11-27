import argparse
import pandas as pd
from sklearn.model_selection import ParameterGrid
from train import train

grid = {
    'quiet': [True],
    'cuda': [0],
    'dataset': ['MovieLens_1M'],
    'split_id': [0],
    'num_negatives': [1, 10, 100],
    'batch_size': [8192],
    'model': ['MF', 'LightGCN'],
    'embed_size': [64],
    'embed_dropout': [0.2],
    'num_layers': [1],
    'edge_dropout': [0.1],
    'loss': ['BCE', 'BPR', 'SCE'],
    'learning_rate': [1e-1, 1e-2],
    'weight_decay': [1e-6, 1e-7, 1e-8],
    'num_epochs': [300],
    'eval_interval': [5],
    'eval_size': [1024],
    'ks': [[5, 20, 100]],
}

combinations, results = list(ParameterGrid(grid)), []
for i, combination in enumerate(combinations, start=1):
    print(f'[Combination {i:3}/{len(combinations)}]', end=' ')
    args = argparse.Namespace(**combination)
    result = train(args)
    results.append(result.values())
columns = result.keys()

pd.DataFrame(results, columns=columns).to_csv(f'log/{"_".join(grid["dataset"] + grid["model"])}_results.csv', index=False)
