import time
import argparse
from data import densify_interactions, get_data
from metric import (
    init_dcg_lists, init_metrics, compute_metrics,
    get_elapsed_time, print_progress, print_result, save_metrics)
import torch

DATASETS = ['Benchmark_Amazon_Books', 'Benchmark_Gowalla', 'Benchmark_Yelp', 'Amazon_Sports', 'MovieLens_1M']
MODELS = ['LinkProp', 'UserKNN++', 'ItemKNN++']


def get_sparse_matrix_batches(dataset, user_degree_exponent, item_degree_exponent, transpose=False):
    users, items = dataset.tensors[0], dataset.tensors[1]
    num_users, num_items, eval_size = dataset.num_users, dataset.num_items, dataset.eval_size

    user_degrees, item_degrees = users.bincount(minlength=num_users), items.bincount(minlength=num_items)
    user_degrees_norm = user_degrees.clamp(min=1) ** -user_degree_exponent
    item_degrees_norm = item_degrees.clamp(min=1) ** -item_degree_exponent

    start, batches = 0, []
    if not transpose:
        for user in range(0, num_users, eval_size):
            stop = start + user_degrees[user:user + eval_size].sum().item()

            indices = torch.stack([users[start:stop] - user, items[start:stop]])
            values = user_degrees_norm[users[start:stop]] * item_degrees_norm[items[start:stop]]
            num_batch_users = len(user_degrees[user:user + eval_size])

            matrix = torch.sparse_coo_tensor(indices, values, [num_batch_users, num_items], device=values.device)
            batches.append(matrix.to_sparse_csr())
            start = stop
    else:
        items, indices = items.sort()
        users = users[indices]

        for item in range(0, num_items, eval_size):
            stop = start + item_degrees[item:item + eval_size].sum().item()

            indices = torch.stack([users[start:stop], items[start:stop] - item])
            values = user_degrees_norm[users[start:stop]] * item_degrees_norm[items[start:stop]]
            num_batch_items = len(item_degrees[item:item + eval_size])

            matrix = torch.sparse_coo_tensor(indices, values, [num_users, num_batch_items], device=values.device).t()
            batches.append(matrix.to_sparse_csr())
            start = stop
    return batches


def get_sparse_matrix(dataset, user_degree_exponent, item_degree_exponent, transpose=False):
    users, items = dataset.tensors[0], dataset.tensors[1]
    num_users, num_items = dataset.num_users, dataset.num_items

    user_degrees_norm = users.bincount(minlength=num_users).clamp(min=1) ** -user_degree_exponent
    item_degrees_norm = items.bincount(minlength=num_items).clamp(min=1) ** -item_degree_exponent

    indices = torch.stack([users, items])
    values = user_degrees_norm[users] * item_degrees_norm[items]

    matrix = torch.sparse_coo_tensor(indices, values, [num_users, num_items], device=values.device)
    return matrix.to_sparse_csr() if not transpose else matrix.t().to_sparse_csr()


def train(args):
    total_start_time = time.perf_counter()
    args.ks.sort()
    print(vars(args))
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    dataset = get_data(args.dataset, args.split_id, 0, args.eval_size, device)

    init_dcg_lists(args.ks, device)
    start_time = time.perf_counter()

    if args.model == 'LinkProp':
        matrix1_batches = get_sparse_matrix_batches(dataset, args.alpha, 0)
        matrix2_t = get_sparse_matrix(dataset, args.gamma, args.beta, transpose=True)
        matrix3_t = get_sparse_matrix(dataset, 0, args.delta, transpose=True)

        def get_y_pred(index):
            return (matrix3_t @ (matrix1_batches[index] @ matrix2_t).to_dense().t()).t()

    elif args.model == 'UserKNN++':
        matrix1_batches = get_sparse_matrix_batches(dataset, args.alpha, 0)
        matrix2_t = get_sparse_matrix(dataset, args.gamma, args.beta, transpose=True)
        matrix3 = get_sparse_matrix(dataset, 0, args.delta)

        neighbor_batches = []
        for matrix1 in matrix1_batches:
            similarity = (matrix1 @ matrix2_t).to_dense()
            topk_values, topk_indices = similarity.topk(args.num_neighbors, sorted=False)

            users1 = torch.arange(len(topk_indices), device=device).repeat_interleave(args.num_neighbors)
            users2 = topk_indices.view(-1)

            indices = torch.stack([users1, users2])
            values = topk_values.view(-1)  # values = (topk_values / topk_values.sum(-1)[:, None]).view(-1)

            neighbors = torch.sparse_coo_tensor(indices, values, similarity.shape, device=device)
            neighbor_batches.append(neighbors.to_sparse_csr())

        def get_y_pred(index):
            return (neighbor_batches[index] @ matrix3).to_dense()

    elif args.model == 'ItemKNN++':
        matrix1_batches = get_sparse_matrix_batches(dataset, args.alpha, 0)
        matrix2_t_batches = get_sparse_matrix_batches(dataset, args.gamma, args.beta, transpose=True)
        matrix3 = get_sparse_matrix(dataset, 0, args.delta)

        topk_values_batches, topk_indices_batches = [], []
        for matrix2_t in matrix2_t_batches:
            similarity = (matrix2_t @ matrix3).to_dense()
            topk_values, topk_indices = similarity.topk(args.num_neighbors, sorted=False)
            topk_values_batches.append(topk_values), topk_indices_batches.append(topk_indices)
        topk_values, topk_indices = torch.cat(topk_values_batches), torch.cat(topk_indices_batches)

        items1 = torch.arange(len(topk_indices), device=device).repeat_interleave(args.num_neighbors)
        items2 = topk_indices.view(-1)

        indices = torch.stack([items1, items2])
        values = topk_values.view(-1)

        neighbors_t = torch.sparse_coo_tensor(indices, values, [dataset.num_items, dataset.num_items], device=device).t().to_sparse_csr()

        def get_y_pred(index):
            return (matrix1_batches[index] @ neighbors_t).to_dense()

    metrics = init_metrics(1)
    for batch_id, user in enumerate(range(0, dataset.num_users, args.eval_size)):
        users = slice(user, user + args.eval_size)
        y_pred = get_y_pred(batch_id)

        for split in ['train', 'valid', 'test']:
            user_degrees = dataset.user_degrees[split][users]
            interaction_batch = dataset.interaction_batches[split][batch_id]

            _, indices = y_pred.topk(args.ks[-1], dim=-1)  # argsort is memory intensive!
            interactions = densify_interactions(*interaction_batch, args.eval_size, dataset.num_items)
            y_sorted = interactions.gather(-1, indices)

            mask = user_degrees > 0
            num_users = (dataset.user_degrees[split] > 0).sum().item()
            compute_metrics(metrics, split, y_sorted[mask], user_degrees[mask], num_users)

            if split in ['train', 'valid']:
                y_pred[interaction_batch] = float('-inf')

    end_time = time.perf_counter()
    elapsed_time = get_elapsed_time(start_time, end_time)

    if not args.quiet:
        print_progress(1, 1, 0, metrics, elapsed_time)
        save_metrics(args, metrics)

    total_end_time = time.perf_counter()
    total_elapsed_time = get_elapsed_time(total_start_time, total_end_time)

    result = print_result(metrics, total_elapsed_time)
    return vars(args) | result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--cuda', default=0, type=int)
    parser.add_argument('--dataset', default='MovieLens_1M', type=str, choices=DATASETS)
    parser.add_argument('--split_id', default=0, type=int)
    parser.add_argument('--model', default='LinkProp', type=str, choices=MODELS)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--beta', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.5, type=float)
    parser.add_argument('--delta', default=0.5, type=float)
    parser.add_argument('--num_neighbors', default=100, type=int)
    parser.add_argument('--eval_size', default=1024, type=int)
    parser.add_argument('--ks', nargs='+', default=[5, 20, 100], type=int)
    args = parser.parse_args()

    result = train(args)
