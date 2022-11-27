import json
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# Dictionary that maps dataset to filename.
DATASETS = {'Amazon_Sports': 'Sports_and_Outdoors.csv', 'MovieLens_1M': 'ratings.dat'}


def preprocess(args):
    dirname = f'data/{args.dataset}/'
    pathname = dirname + DATASETS[args.dataset]

    # Load dataset from file.
    if args.dataset == 'Amazon_Sports':
        df = pd.read_csv(pathname, names=['item', 'user', 'rating'], usecols=[0, 1, 2])
    elif args.dataset == 'MovieLens_1M':
        df = pd.read_table(pathname, sep='::', names=['user', 'item', 'rating'], usecols=[0, 1, 2], engine='python')
    df = df.dropna()

    # Filter out unnecessary interactions.
    if 'rating' in df.columns:
        if args.positive_only:
            rating_min, rating_max = df['rating'].min(), df['rating'].max()
            df['rating'] = (df['rating'] - rating_min) / (rating_max - rating_min)
            df = df[df['rating'] > 0.7]
        df = df[['user', 'item']]
    df = df.drop_duplicates()

    # Get k-core of the user-item interaction graph.
    num_interactions = 0
    while num_interactions != len(df):
        num_interactions = len(df)
        user_degrees = df['user'].value_counts(sort=False)
        users = user_degrees.index[user_degrees >= args.k_core]
        df = df[df['user'].isin(users)]

        item_degrees = df['item'].value_counts(sort=False)
        items = item_degrees.index[item_degrees >= args.k_core]
        df = df[df['item'].isin(items)]

    # Map users and items to ids (0, 1, ...).
    df['user'] = df['user'].astype('category').cat.codes
    df['item'] = df['item'].astype('category').cat.codes

    # Save dataset statistics to file.
    num_users = int(df['user'].max() + 1)
    num_items = int(df['item'].max() + 1)
    statistics = {
        'num_users': num_users, 'num_items': num_items,
        'num_interactions': num_interactions,
        'density': num_interactions / (num_users * num_items),
        'median_user_degrees': user_degrees.median(), 'median_item_degrees': item_degrees.median()}
    with open(dirname + 'statistics.json', 'w') as f:
        json.dump(statistics, f, indent=4)

    # Split dataset into train, valid, and test.
    for split_id in range(args.num_splits):
        df_train, df_test = train_test_split(df, test_size=args.test_size, random_state=split_id, stratify=df['user'])
        df_train, df_valid = train_test_split(df_train, test_size=args.valid_size, random_state=split_id, stratify=df_train['user'])

        # Save dataset splits to file.
        df_train.to_csv(dirname + f'train_{split_id}.csv', index=False)
        df_valid.to_csv(dirname + f'valid_{split_id}.csv', index=False)
        df_test.to_csv(dirname + f'test_{split_id}.csv', index=False)

    with open(dirname + 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MovieLens_1M', type=str, choices=DATASETS.keys())
    parser.add_argument('--positive_only', action='store_true')
    parser.add_argument('--k_core', default=10, type=int)
    parser.add_argument('--num_splits', default=1, type=int)
    parser.add_argument('--valid_size', default=0.125, type=float)
    parser.add_argument('--test_size', default=0.2, type=float)
    args = parser.parse_args()

    preprocess(args)
