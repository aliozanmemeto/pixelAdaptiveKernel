import csv
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image

import torch
from skimage import io


# ===================================================== DATALOADING ====================================================
def read_split(path: Path) -> List[str]:
    with open(path) as f:
        lines = f.read().splitlines()
    return lines


def load_img(path: Path) -> torch.tensor:
    return torch.from_numpy(io.imread(path)).permute(2, 0, 1) / 255.0


def load_annotation(path: Path) -> torch.Tensor:
    img = io.imread(path)
    if img.ndim == 3:
        img = img[..., 0]
    return torch.from_numpy(img) / 255.0


def save_annotation(path: Path, annotation: torch.Tensor) -> None:
    img = (annotation * 255.0).detach().cpu().numpy().astype("uint8")
    io.imsave(path, img)


def load_semantic_annotation(path: Path) -> torch.Tensor:
    img = io.imread(path)
    return torch.from_numpy(img).permute(2, 0, 1)


def load_instance_annotation(path: Path) -> torch.Tensor:
    img = io.imread(path)
    return torch.from_numpy(img).permute(2, 0, 1)


def combine_annotation(
    semantic_annotation: torch.Tensor, panoptic_annotation: torch.Tensor
) -> torch.Tensor:
    return semantic_annotation


def load_feature_map(path: Path) -> torch.Tensor:
    f_map = torch.load(path).to(torch.float32)
    return f_map


def read_class_dict(path: Path) -> Dict[Tuple[int, int, int], Tuple[int, str]]:
    names = []
    colors = []
    class_dict = {}
    with open(path.joinpath("reduced_class_dict.csv"), "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        # skip header
        next(csv_reader, None)
        for line in csv_reader:
            names.append(line[0])
            colors.append((int(line[1]), int(line[2]), int(line[3])))

    for i, (name, color) in enumerate(zip(names, colors)):
        class_dict[color] = (i, name)

    return class_dict


# ==================================================== MISCELLANEOUS ===================================================
def colors_to_labels(
    color_annotation: torch.Tensor,
    class_dict: Dict[Tuple[int, int, int], Tuple[int, str]],
) -> torch.Tensor:
    label_annotation = -torch.ones_like(color_annotation)[0]
    for color, (label, name) in class_dict.items():
        mask = torch.all(
            color_annotation
            == torch.tensor([color[0], color[1], color[2]])[:, None, None],
            dim=0,
        )
        label_annotation[mask] = label
    return label_annotation


def labels_to_color(
    label_annotation: torch.Tensor,
    class_dict: Dict[Tuple[int, int, int], Tuple[int, str]],
) -> torch.Tensor:
    color_annotation = -torch.ones(label_annotation.shape + (3,))
    for color, (label, name) in class_dict.items():
        mask = label_annotation == label
        color_annotation[mask] = torch.tensor(
            [color[0], color[1], color[2]], dtype=torch.float32
        )
    return color_annotation.permute(2, 0, 1)


def metrics_header():
    print(
        f"{f'Epoch' : >8} {f'Split' : >10} {f'Loss' : >6} {f'Acc' : >5} {f'Prcn' : >5} {f'Rcll' : >5} {f'IOU' : >5}"
    )


def print_metrics(metrics: Dict[str, float], epoch: int, split_name: str):
    metric_names = ["loss", "acc", "m_prcn", "m_rcll", "m_iou"]
    str_metrics: Dict[str, str] = {}
    for metric in metric_names:
        value = metrics.get(metric, None)
        if value:
            str_metrics[metric] = f"{value :.2f}"
        else:
            str_metrics[metric] = "-"

    loss = str_metrics["loss"]
    acc = str_metrics["acc"]
    m_prcn = str_metrics["m_prcn"]
    m_rcll = str_metrics["m_rcll"]
    m_iou = str_metrics["m_iou"]

    print(
        f"{f'{epoch}' : >8} {f'{split_name}' : >10} {f'{loss}' : >6} {f'{acc}' : >5} {f'{m_prcn}' : >5} {f'{m_rcll}' : >5} {f'{m_iou}' : >5}"
    )


def feature_map_pca(feature_map: torch.Tensor) -> torch.Tensor:
    q_estimate = min([feature_map.shape[0], feature_map.shape[1], 100])
    U, S, V = torch.pca_lowrank(feature_map, q=q_estimate, center=True, niter=100)
    pca_map = feature_map @ V[:, :3]
    pca_min = torch.min(pca_map, dim=0, keepdim=True)[0]
    pca_max = torch.max(pca_map, dim=0, keepdim=True)[0]
    return (pca_map - pca_min) / (pca_max - pca_min)


# ===================================================== PREDICTION =====================================================
def logits_to_labels(logits: torch.Tensor) -> torch.Tensor:
    labels = logits.max(1).indices
    return labels


def binary_output_to_labels(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.zeros_like(logits)
    labels[logits >= 0.5] = 1.0
    return labels
