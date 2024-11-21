import torch


def pairwise_distance(
    x: torch.Tensor, y: torch.Tensor, normalized: bool = False, channel_first: bool = False
) -> torch.Tensor:
    r"""Pairwise distance of two (batched) point clouds.

    Args:
        x (Tensor): (*, N, C) or (*, C, N)
        y (Tensor): (*, M, C) or (*, C, M)
        normalized (bool=False): if the points are normalized, we have "x2 + y2 = 1", so "d2 = 2 - 2xy".
        channel_first (bool=False): if True, the points shape is (*, C, N).

    Returns:
        dist: torch.Tensor (*, N, M)
    """
    if channel_first:
        channel_dim = -2
        xy = torch.matmul(x.transpose(-1, -2), y)  # [(*, C, N) -> (*, N, C)] x (*, C, M)
    else:
        channel_dim = -1
        xy = torch.matmul(x, y.transpose(-1, -2))  # (*, N, C) x [(*, M, C) -> (*, C, M)]
    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = torch.sum(x ** 2, dim=channel_dim).unsqueeze(-1)  # (*, N, C) or (*, C, N) -> (*, N) -> (*, N, 1)
        y2 = torch.sum(y ** 2, dim=channel_dim).unsqueeze(-2)  # (*, M, C) or (*, C, M) -> (*, M) -> (*, 1, M)
        sq_distances = x2 - 2 * xy + y2
    sq_distances = sq_distances.clamp(min=1e-12)
    return sq_distances

import numpy as np
def pairwise_distance_np(x, y):
    r"""Pairwise distance of two (batched) point clouds.

    Args:

    Returns:

    """
    # transpose dim 1 and 2


    xy = np.matmul(x, y.swapaxes(-1, -2))  # (*, N, C) x [(*, M, C) -> (*, C, M)]

    x2 = np.expand_dims(np.sum(x ** 2, axis=-1),-1)  # (*, N, C) or (*, C, N) -> (*, N) -> (*, N, 1)
    y2 = np.expand_dims(np.sum(y ** 2, axis=-1),-2)  # (*, M, C) or (*, C, M) -> (*, M) -> (*, 1, M)
    sq_distances = x2 - 2 * xy + y2
    sq_distances[sq_distances<0]=1e-12
    return sq_distances
