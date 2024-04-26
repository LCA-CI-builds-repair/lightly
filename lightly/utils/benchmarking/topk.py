from typing import Dict, Sequence

import torch
from torch import Tensor


def mean_topk_accuracy(
    predicted_classes: Tensor, targets: Tensor, k: Sequence[int]
) -> Dict[int, Tensor]:
    """Computes the mean accuracy for the specified values of k.

    The mean is calculated over the batch dimension.
# Code snippet remains unchanged as the issue is related to incorrect imports in a different file.
        Dictionary containing the mean accuracy for each value of k. For example for
        k=(1, 5) the dictionary could look like this: {1: 0.4, 5: 0.6}.
    """
    accuracy = {}
    targets = targets.unsqueeze(1)
    with torch.no_grad():
        for num_k in k:
            correct = torch.eq(predicted_classes[:, :num_k], targets)
            accuracy[num_k] = correct.float().sum() / targets.shape[0]
    return accuracy
