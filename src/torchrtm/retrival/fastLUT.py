import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
from torch.nn import functional as F

def call_torch(xb, xq, k, distance_order=2):
    """
    Computes the pairwise distances between query points (xq) and reference points (xb)
    using the specified Minkowski distance (p-norm). Returns the top-k nearest neighbors.

    Args:
        xb (torch.Tensor): Reference points (database).
        xq (torch.Tensor): Query points.
        k (int): Number of nearest neighbors to retrieve.
        distance_order (int): Order of the Minkowski distance (default is Euclidean distance, p=2).

    Returns:
        torch.Tensor: Indices of the k-nearest neighbors for each query point.
    """
    bs_scores = torch.cdist(xq, xb, p=distance_order)  # Compute distance matrix
    return bs_scores.topk(k=k, dim=1, sorted=True, largest=False)  # Retrieve top-k nearest neighbors

def call_torch_batched(xb, xq, k, distance_order=2, batch_size=64):
    """
    Computes the k-nearest neighbors in batches to handle large datasets efficiently.

    Returns:
        torch.Tensor: Indices of the k-nearest neighbors for each query point.
    """
    n = xq.shape[0]
    topk_indices = []
    for i in range(0, n, batch_size):
        xq_batch = xq[i:i+batch_size]  # Extract batch
        bs_scores = torch.cdist(xq_batch, xb, p=distance_order)  # Compute distances
        _, indices = bs_scores.topk(k=k, dim=1, sorted=True, largest=False)  # Get top-k indices
        topk_indices.append(indices)
    print('bs_scores')
    return torch.cat(topk_indices, dim=0)

def Torchlut_pred(xb, xq, k, y, distance_order=1, batch_size=1000):
    """
    Performs k-nearest neighbors lookup and predicts values based on the nearest neighbors.

    Args:
        xb (torch.Tensor): Reference points (database).
        xq (torch.Tensor): Query points.
        k (int): Number of nearest neighbors to consider.
        y (torch.Tensor): Target values corresponding to reference points.
        distance_order (int): Order of the Minkowski distance.
        batch_size (int): Size of each batch (used for batched processing).

    Returns:
        torch.Tensor: Predicted values based on the mean of k-nearest neighbors.
    """
    if len(xq) < batch_size:
        index_table = call_torch(xb, xq, k, distance_order)[:][1]  # Get nearest neighbor indices
    else:
        print('Start batch processing...')
        index_table = call_torch_batched(xb, xq, k, distance_order, batch_size=batch_size)[:] # Batched nearest neighbor search
    print(index_table.shape)
    selected_values = y[index_table]  # Retrieve target values for the nearest neighbors
    #print(selected_values.shape)
    pred = selected_values.mean(dim=1)  # Compute the mean value across k-nearest neighbors

    return pred