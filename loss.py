import torch
import torch.nn.functional as F


def bce_loss(y_pred, y_neg_pred):
    y = torch.ones_like(y_pred)  # Shape: (batch_size,)
    y_neg = torch.zeros_like(y_neg_pred)  # Shape: (batch_size, num_negatives)

    num_negatives = y_neg.shape[-1]
    loss = F.binary_cross_entropy_with_logits(y_pred, y)
    loss += F.binary_cross_entropy_with_logits(y_neg_pred, y_neg) * num_negatives
    return loss / (num_negatives + 1)


def bpr_loss(y_pred, y_neg_pred):
    y_pred = y_pred[:, None] - y_neg_pred
    y = torch.ones_like(y_pred)  # Shape: (batch_size, num_negatives)

    loss = F.binary_cross_entropy_with_logits(y_pred, y)
    return loss


def sce_loss(y_pred, y_neg_pred):
    y_pred = torch.cat([y_pred[:, None], y_neg_pred], -1)  # Shape: (batch_size, num_negatives + 1)
    y = torch.zeros(len(y_pred), dtype=torch.long, device=y_pred.device)

    loss = F.cross_entropy(y_pred, y)
    return loss
