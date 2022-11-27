import argparse
import pandas as pd
from sklearn.model_selection import ParameterGrid
from train_knn import train

grid = {
    'quiet': [True],
    'cuda': [0],
    'dataset': ['MovieLens_1M'],
    'split_id': [0],
    'model': ['UserKNN++', 'ItemKNN++'],
    'alpha': [0.5],
    'beta': [0.1, 0.3, 0.5, 0.7, 0.9],
    'gamma': [0.1, 0.3, 0.5, 0.7, 0.9],
    'delta': [0.1, 0.3, 0.5, 0.7, 0.9],
    'num_neighbors': [10, 100, 300],
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
