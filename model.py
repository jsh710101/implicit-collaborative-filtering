import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


def dot_product(users, items, items_neg):
    if items_neg is not None:  # If the model is in training mode.
        items = (users * items).sum(-1)  # Shape: (batch_size,)
        items_neg = (users[:, None] * items_neg).sum(-1)  # Shape: (batch_size, num_negatives)
    else:  # If the model is in evaluation mode.
        items = users @ items.t()  # Shape: (eval_size, num_items)
    return items, items_neg


class MF(nn.Module):
    def __init__(self, dataset, embed_size, embed_dropout, *args):
        super().__init__()
        self.user_embeds = Parameter(torch.randn(dataset.num_users, embed_size))
        self.item_embeds = Parameter(torch.randn(dataset.num_items, embed_size))
        self.dropout = nn.Dropout(embed_dropout)

    def drop_edges(self, dataset):
        pass

    def compute_final_embeds(self):
        self.final_user_embeds = self.user_embeds  # Shape: (num_users, embed_size)
        self.final_item_embeds = self.item_embeds  # Shape: (num_items, embed_size)

    def forward(self, users, items, items_neg=None):
        if items_neg is not None:  # If the model is in training mode.
            self.compute_final_embeds()
            items_neg = self.dropout(self.final_item_embeds[items_neg])  # Shape: (batch_size, num_negatives, embed_size)

        users = self.dropout(self.final_user_embeds[users])  # Shape: (batch_size, embed_size) for training, (eval_size, embed_size) for evaluation
        items = self.dropout(self.final_item_embeds[items])  # Shape: (batch_size, embed_size) for training, (num_items, embed_size) for evaluation
        return dot_product(users, items, items_neg)


class LightGCN(nn.Module):
    def __init__(self, dataset, embed_size, embed_dropout, num_layers, edge_dropout, *args):
        super().__init__()
        self.user_embeds = Parameter(torch.randn(dataset.num_users, embed_size))
        self.item_embeds = Parameter(torch.randn(dataset.num_items, embed_size))
        self.dropout = nn.Dropout(embed_dropout)

        # Precompute the transpose of the sparse matrix for efficiency.
        self.matrix, self.matrix_t = self.get_sparse_matrix(
            dataset.tensors[0], dataset.tensors[1], dataset.num_users, dataset.num_items)
        self.num_layers, self.edge_dropout = num_layers, edge_dropout

    def get_sparse_matrix(self, users, items, num_users, num_items):
        user_degrees_norm = users.bincount(minlength=num_users).clamp(min=1) ** -0.5
        item_degrees_norm = items.bincount(minlength=num_items).clamp(min=1) ** -0.5

        indices = torch.stack([users, items])
        values = user_degrees_norm[users] * item_degrees_norm[items]

        matrix = torch.sparse_coo_tensor(indices, values, [num_users, num_items], device=values.device)
        # There is a bug that PyTorch returns the transpose of a sparse csr tensor always in cuda:0.
        return matrix.to_sparse_csr(), matrix.t().to_sparse_csr()  # Speed: csr > coalesced coo > coo

    def drop_edges(self, dataset):  # Implemented based on DropEdge (https://arxiv.org/abs/1907.10903).
        if self.edge_dropout > 0:
            users, items = dataset.tensors[0], dataset.tensors[1]
            mask = torch.rand_like(users, dtype=torch.float) > self.edge_dropout

            self.matrix_drop, self.matrix_drop_t = self.get_sparse_matrix(
                users[mask], items[mask], dataset.num_users, dataset.num_items)

    def compute_final_embeds(self):
        do_drop_edges = self.training and self.edge_dropout > 0
        matrix, matrix_t = (self.matrix_drop, self.matrix_drop_t) if do_drop_edges else (self.matrix, self.matrix_t)

        user_embeds_list, item_embeds_list = [self.user_embeds], [self.item_embeds]
        for i in range(self.num_layers):
            user_embeds_list.append(matrix @ item_embeds_list[i])
            item_embeds_list.append(matrix_t @ user_embeds_list[i])

        self.final_user_embeds = sum(user_embeds_list) / len(user_embeds_list)
        self.final_item_embeds = sum(item_embeds_list) / len(item_embeds_list)

    def forward(self, users, items, items_neg=None):
        if items_neg is not None:  # If the model is in training mode.
            self.compute_final_embeds()
            items_neg = self.dropout(self.final_item_embeds[items_neg])  # Shape: (batch_size, num_negatives, embed_size)

        users = self.dropout(self.final_user_embeds[users])  # Shape: (batch_size, embed_size) for training, (eval_size, embed_size) for evaluation
        items = self.dropout(self.final_item_embeds[items])  # Shape: (batch_size, embed_size) for training, (num_items, embed_size) for evaluation
        return dot_product(users, items, items_neg)
