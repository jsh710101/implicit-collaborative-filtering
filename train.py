import time
import math
import argparse
from data import densify_interactions, get_data, sample_negatives
from model import MF, LightGCN
from loss import bce_loss, bpr_loss, sce_loss
from metric import (
    init_dcg_lists, init_metrics, compute_metrics, update_best_metrics,
    get_elapsed_time, print_progress, print_result, save_metrics)
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

DATASETS = ['Benchmark_Amazon_Books', 'Benchmark_Gowalla', 'Benchmark_Yelp', 'Amazon_Sports', 'MovieLens_1M']
MODELS = {'MF': MF, 'LightGCN': LightGCN}
LOSSES = {'BCE': bce_loss, 'BPR': bpr_loss, 'SCE': sce_loss}


def train(args):
    total_start_time = time.perf_counter()
    args.ks.sort()
    print(vars(args))
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    dataset = get_data(args.dataset, args.split_id, args.num_negatives, args.eval_size, device)
    model = MODELS[args.model](
        dataset, args.embed_size, args.embed_dropout, args.num_layers, args.edge_dropout).to(device)

    compute_loss = LOSSES[args.loss]
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.num_epochs)

    init_dcg_lists(args.ks, device)
    best_metrics = init_metrics()

    for epoch in range(1, args.num_epochs + 1):
        start_time = time.perf_counter()
        sample_negatives(dataset)
        model.drop_edges(dataset)

        model.train()
        train_loss = 0

        index_batches = torch.randperm(len(dataset), device=device).split(args.batch_size)
        for indices in index_batches:  # Faster than DataLoader.
            users, items, items_neg = [tensor[indices] for tensor in dataset.tensors]
            y_pred, y_neg_pred = model(users, items, items_neg)
            loss = compute_loss(y_pred, y_neg_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(indices) / len(dataset)

        if math.isnan(train_loss):  # Sanity check.
            break

        if epoch % args.eval_interval == 0:
            model.eval()
            metrics = init_metrics(epoch)

            with torch.no_grad():
                model.compute_final_embeds()

                for batch_id, user in enumerate(range(0, dataset.num_users, args.eval_size)):
                    users, items = slice(user, user + args.eval_size), slice(None)
                    y_pred, y_neg_pred = model(users, items)

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
                update_best_metrics(best_metrics, metrics, model, args.quiet)
        else:
            metrics = None

        scheduler.step()
        end_time = time.perf_counter()
        elapsed_time = get_elapsed_time(start_time, end_time)

        if not args.quiet:
            print_progress(args.num_epochs, epoch, train_loss, metrics, elapsed_time)

    total_end_time = time.perf_counter()
    total_elapsed_time = get_elapsed_time(total_start_time, total_end_time)

    result = print_result(best_metrics, total_elapsed_time)
    if not args.quiet:
        save_metrics(args, best_metrics)

    return vars(args) | result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--cuda', default=0, type=int)
    parser.add_argument('--dataset', default='MovieLens_1M', type=str, choices=DATASETS)
    parser.add_argument('--split_id', default=0, type=int)
    parser.add_argument('--num_negatives', default=1, type=int)
    parser.add_argument('--batch_size', default=8192, type=int)
    parser.add_argument('--model', default='MF', type=str, choices=MODELS.keys())
    parser.add_argument('--embed_size', default=64, type=int)
    parser.add_argument('--embed_dropout', default=0.2, type=float)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--edge_dropout', default=0.1, type=float)
    parser.add_argument('--loss', default='BPR', type=str, choices=LOSSES.keys())
    parser.add_argument('--learning_rate', default=1e-2, type=float)
    parser.add_argument('--weight_decay', default=1e-7, type=float)
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--eval_interval', default=5, type=int)
    parser.add_argument('--eval_size', default=1024, type=int)
    parser.add_argument('--ks', nargs='+', default=[5, 20, 100], type=int)
    args = parser.parse_args()

    result = train(args)
