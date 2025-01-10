from typing import List, Optional, Tuple

import torch
from sklearn.metrics import confusion_matrix as sk_conf_matrix


def pixelwise_accuracy(prediction: torch.Tensor, annotations: torch.Tensor) -> Tuple[int, int]:
    # prediction: B,H,W
    # annotations: B,H,W
    pred_correct = int(((prediction - annotations) == 0).sum().item())
    total = int(prediction.view(-1).shape[0])

    return pred_correct, total


def confusion_matrix(prediction: torch.Tensor, annotations: torch.Tensor, num_classes: int) -> torch.Tensor:
    conf_matrix = torch.zeros(num_classes, num_classes)

    cm = torch.from_numpy(sk_conf_matrix(prediction.reshape(-1).cpu(), annotations.reshape(-1).cpu()))
    conf_matrix[: cm.shape[0], : cm.shape[1]] = cm

    return conf_matrix


def precision(true_positive: int, false_positive: int) -> float:
    return (true_positive) / (true_positive + false_positive)


def recall(true_positive: int, false_negative: int) -> float:
    return (true_positive) / (true_positive + false_negative)


def accuracy(conf_matrix: torch.Tensor) -> float:
    return (torch.trace(conf_matrix) / torch.sum(conf_matrix)).item()


def mean_precision(conf_matrix: torch.Tensor, mask: Optional[List] = None):
    precisions = []
    for idx, row in enumerate(conf_matrix):
        if mask is not None:
            if mask[idx] == 0:
                continue
        precisions.append((row[idx] / row.sum()).item())

    return sum(precisions) / len(precisions)


def mean_recall(conf_matrix: torch.Tensor, mask: Optional[List] = None):
    recalls = []
    for idx, col in enumerate(conf_matrix.transpose(0, 1)):
        if mask is not None:
            if mask[idx] == 0:
                continue
        recalls.append((col[idx] / col.sum()).item())

    return sum(recalls) / len(recalls)


def mean_iou(conf_matrix: torch.Tensor, mask: Optional[List] = None):
    ious = []
    for idx in range(conf_matrix.shape[0]):
        if mask is not None:
            if mask[idx] == 0:
                continue
        ious.append(
            (
                conf_matrix[idx, idx] / (conf_matrix[idx, :].sum() + conf_matrix[:, idx].sum() - conf_matrix[idx, idx])
            ).sum()
        )

    return sum(ious) / len(ious)
