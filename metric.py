import copy
import json
import torch

KS = None
DCG_LIST, IDCG_LIST = None, None


def precision_k(y, k, y_sum):
    return y[:, :k].sum(-1) / k


def recall_k(y, k, y_sum):
    return y[:, :k].sum(-1) / y_sum


def normalized_recall_k(y, k, y_sum):
    return y[:, :k].sum(-1) / y_sum.clamp(max=k)


def ndcg_k(y, k, y_sum):
    dcg_k = (DCG_LIST[:k] * y[:, :k]).sum(-1)
    idcg_k = IDCG_LIST[y_sum.clamp(max=k) - 1]
    return dcg_k / idcg_k


METRICS = {'Precision': precision_k, 'Recall': recall_k, 'NDCG': ndcg_k}


def init_dcg_lists(ks, device):
    global KS, DCG_LIST, IDCG_LIST
    KS = ks

    # `DCG_LIST[i - 1] = 1 / log2(i + 1)`
    DCG_LIST = 1 / torch.log2(torch.arange(KS[-1], device=device) + 2)
    # `IDCG_LIST[i - 1] = DCG_LIST[0] + ... + DCG_LIST[i - 1]'
    IDCG_LIST = DCG_LIST.cumsum(0)


def init_metrics(epoch=0):
    metrics = {'model': None}
    for metric in METRICS:
        metrics[metric] = {k: {'epoch': epoch} | {split: 0 for split in ['train', 'valid', 'test']} for k in KS}
    return metrics


def compute_metrics(metrics, split, y, y_sum, num_users):
    for metric, compute_metric in METRICS.items():
        for k in KS:
            metrics[metric][k][split] += compute_metric(y, k, y_sum).sum().item() / num_users


def update_best_metrics(best_metrics, metrics, model, quiet):
    for metric in METRICS:
        for k in KS:
            if best_metrics[metric][k]['valid'] <= metrics[metric][k]['valid']:
                best_metrics[metric][k] = metrics[metric][k]

                if metric == 'NDCG' and k == KS[-1] and not quiet:
                    best_metrics['model'] = copy.deepcopy(model.state_dict())


def get_elapsed_time(start, end):
    seconds = end - start
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f'{hours:02}:{minutes:02}:{seconds:06.3f}'


def print_progress(num_epochs, epoch, train_loss, metrics, elapsed_time):
    print(f'[Epoch {epoch:3}/{num_epochs}]', end=' ')
    print(f'Loss: {train_loss:6.4f}', end=' | ')
    print(elapsed_time)

    if metrics is not None:
        for metric in METRICS:
            print(end=' ' * 8)
            for k in KS:
                print(f'{metric:>10}_{k}:', end=' ')
                for split in ['train', 'valid', 'test']:
                    print(f'{metrics[metric][k][split]:6.4f}', end=' ')
                print(end='|')
            print()


def print_result(metrics, elapsed_time):
    print('[Best Checkpoint]', end=' ')
    print(elapsed_time)

    for metric in METRICS:
        print(end=' ' * 8)
        for k in KS:
            print(f'{metric:>10}_{k}: {metrics[metric][k]["test"]:6.4f} ({metrics[metric][k]["epoch"]:3})', end=' |')
        print()

    result = {}
    for metric in METRICS:
        for k in KS:
            for split in ['train', 'valid', 'test']:
                result[f'{metric}_{k}_{split}'] = metrics[metric][k][split]

    return result


def save_metrics(args, metrics):
    with open(f'model/{args.dataset}_{args.model}_{args.split_id}_args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    torch.save(metrics['model'], f'model/{args.dataset}_{args.model}_{args.split_id}.pt')

    del metrics['model']
    with open(f'log/{args.dataset}_{args.model}_{args.split_id}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
