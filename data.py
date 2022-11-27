import json
import pandas as pd
import torch
from torch.utils.data import TensorDataset


def get_nums(dirname):
    with open(dirname + 'statistics.json', 'r') as f:
        statistics = json.load(f)
    return statistics['num_users'], statistics['num_items']


def batchfy_interactions(users, items, user_degrees, eval_size):
    # Let `matrix` be the user-item interaction matrix of shape (num_users, num_items).
    # ith batch `batches[i]` contains the indices of observed interactions in `matrix[i * eval_size : (i+1) * eval_size, :]`.
    batches = []
    start, num_users = 0, len(user_degrees)
    for user in range(0, num_users, eval_size):
        stop = start + user_degrees[user:user + eval_size].sum().item()
        batches.append([users[start:stop] - user, items[start:stop]])
        start = stop
    return batches


def densify_interactions(users, items, eval_size, num_items):
    interactions = torch.zeros([eval_size, num_items], dtype=torch.bool, device=users.device)
    interactions[users, items] = True
    return interactions


def get_data(dataset, split_id, num_negatives, eval_size, device):
    dirname = f'data/{dataset}/'
    num_users, num_items = get_nums(dirname)

    for split in ['train', 'valid', 'test']:
        df = pd.read_csv(dirname + f'{split}_{split_id}.csv')
        users = torch.tensor(df['user'].to_numpy(), device=device)
        items = torch.tensor(df['item'].to_numpy(), device=device)

        # Sort interactions by user ids.
        users, indices = users.sort()
        items = items[indices]

        if split == 'train':
            items_neg = items[:, None].repeat(1, num_negatives)  # Shape: (num_interactions, num_negatives)
            # Initialize dataset object.
            dataset = TensorDataset(users, items, items_neg)
            dataset.num_users, dataset.num_items, dataset.eval_size = num_users, num_items, eval_size
            dataset.user_degrees, dataset.interaction_batches = {}, {}
        dataset.user_degrees[split] = users.bincount(minlength=num_users)
        dataset.interaction_batches[split] = batchfy_interactions(users, items, dataset.user_degrees[split], eval_size)
    return dataset


def sample_negatives(dataset):  # Note that this function samples negatives with replacement for efficiency.
    dataset.tensors[2].random_(dataset.num_items)

    start = 0
    for users, items in dataset.interaction_batches['train']:
        stop = start + len(users)
        items_neg = dataset.tensors[2][start:stop]  # Slicing creates a view, not a copy.
        interactions = densify_interactions(users, items, dataset.eval_size, dataset.num_items)
        start = stop

        mask = interactions[users[:, None], items_neg]  # Indicates negative smaples that should be resampled.
        while mask.any():
            items_neg[mask] = torch.randint(dataset.num_items, (mask.sum(),), device=items_neg.device)
            mask = interactions[users[:, None], items_neg]
