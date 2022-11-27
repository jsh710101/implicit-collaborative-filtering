import json
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

DATASETS = ['Amazon_Books', 'Gowalla', 'Yelp']


def preprocess(args):
    dirname = f'data/Benchmark_{args.dataset}/'

    num_users, num_items, num_interactions = 0, 0, 0
    for split in ['train', 'test']:
        # Load dataset from file.
        with open(dirname + f'{split}.txt', 'r') as f:
            lines = f.readlines()

        # Convert each line to interactions.
        interactions = []
        for line in lines:
            user, *items = [int(elem) for elem in line.split()]
            for item in items:
                interactions.append([user, item])

        # Save dataset splits to file.
        df = pd.DataFrame(interactions, columns=['user', 'item'])
        df.to_csv(dirname + f'{split}_0.csv', index=False)
        if split == 'train':
            df_train, df_valid = train_test_split(df, test_size=args.valid_size, random_state=0, stratify=df['user'])
            df_train.to_csv(dirname + 'train_0.csv', index=False)
            df_valid.to_csv(dirname + 'valid_0.csv', index=False)

        num_users = max(num_users, df['user'].max() + 1)
        num_items = max(num_items, df['item'].max() + 1)
        num_interactions += len(df)

    # Save dataset statistics to file.
    statistics = {
        'num_users': num_users, 'num_items': num_items,
        'num_interactions': num_interactions,
        'density': num_interactions / (num_users * num_items)}
    with open(dirname + 'statistics.json', 'w') as f:
        json.dump(statistics, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Amazon_Books', type=str, choices=DATASETS)
    parser.add_argument('--valid_size', default=0.1, type=float)
    args = parser.parse_args()

    preprocess(args)
