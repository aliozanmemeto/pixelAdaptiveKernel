from typing import Dict, Union
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from exercise_code.data.obj_segmentation.pac_network import PACNet
from exercise_code.data.obj_segmentation.linear_probing import LinearProbingNet
from exercise_code.data.seg_datasets.davis_obj_seg import DavisDataset
from exercise_code.data.utils import (
    print_metrics,
    metrics_header,
    binary_output_to_labels,
)
from exercise_code.data.metrics import (
    accuracy,
    mean_precision,
    mean_recall,
    mean_iou,
    confusion_matrix,
)

dpi = 96


def training(
    model: Union[LinearProbingNet, PACNet],
    dataloaders: Dict[str, DataLoader],
    loss_fn,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
):
    num_classes = 2

    def _train_epoch(dataloader: DataLoader) -> Dict[str, float]:
        epoch_loss = 0
        num_datapoints = 0
        model.train()
        for batch_idx, data in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)
            feature_maps = data["feature_map"].detach().to(device)
            annotation_kw = "annotations"
            if type(model) == LinearProbingNet:
                annotation_kw = "annotations_coarse"
            annotations = data[annotation_kw].to(device)
            kwargs = {"guide": data["image"].to(device)}

            segmentation = model(feature_maps.permute(0, 3, 1, 2), **kwargs)

            loss = loss_fn(segmentation[:, 0, ...], annotations)
            epoch_loss += loss.detach().cpu().item() * feature_maps.shape[0]
            num_datapoints += feature_maps.shape[0]

            loss.backward()
            optimizer.step()

        metrics: Dict[str, float] = {}
        metrics["loss"] = epoch_loss / num_datapoints
        return metrics

    def _eval_epoch(
        dataloader: DataLoader, full_resolution: bool = False
    ) -> Dict[str, Dict[str, float]]:
        epoch_loss = 0
        num_datapoints = 0
        model.eval()

        category_names = dataloader.dataset.category_names
        conf_matrix = {
            category_name: torch.zeros(num_classes, num_classes)
            for category_name in category_names
        }
        conf_matrix["overall"] = torch.zeros(num_classes, num_classes)
        for data in dataloader:
            feature_maps = data["feature_map"].to(device)
            annotation_kw = "annotations"
            if type(model) == LinearProbingNet and not full_resolution:
                annotation_kw = "annotations_coarse"
            annotations = data[annotation_kw].to(device)
            kwargs = {"guide": data["image"].to(device)}
            categories = data["category"]

            segmentation = model(feature_maps.permute(0, 3, 1, 2), **kwargs)

            if full_resolution and type(model) == LinearProbingNet:
                segmentation = F.interpolate(
                    segmentation, annotations.shape[-2:], mode="nearest"
                )

            loss = loss_fn(segmentation[:, 0, ...], annotations)
            epoch_loss += loss.detach().cpu().item() * feature_maps.shape[0]
            num_datapoints += feature_maps.shape[0]

            labels = binary_output_to_labels(segmentation[:, 0, ...])

            for category in set(categories):
                indices = [i for i, x in enumerate(categories) if x == category]
                result = (
                    confusion_matrix(labels[indices], annotations[indices], num_classes)
                    .detach()
                    .cpu()
                )
                conf_matrix[category] = conf_matrix[category] + result
                conf_matrix["overall"] = conf_matrix["overall"] + result
            # conf_matrix = conf_matrix + confusion_matrix(labels, annotations, num_classes).detach().cpu()

        metrics: Dict[str, Dict[str, float]] = {
            category: {} for category in category_names
        }
        metrics["overall"] = {}
        for category, matrix in conf_matrix.items():
            category_metrics: Dict[str, float] = {}
            category_metrics["loss"] = epoch_loss / num_datapoints
            category_metrics["acc"] = accuracy(conf_matrix[category])
            category_metrics["m_prcn"] = mean_precision(conf_matrix[category])
            category_metrics["m_rcll"] = mean_recall(conf_matrix[category])
            category_metrics["m_iou"] = mean_iou(conf_matrix[category])

            metrics[category] = category_metrics

        return metrics

    metrics_header()
    for epoch in range(1, num_epochs + 1):
        # print(epoch)
        results = _train_epoch(dataloaders["train"])
        print_metrics(results, epoch, "train")
        if epoch % 5 == 0 or epoch == num_epochs:
            if epoch != num_epochs:
                metrics_val = _eval_epoch(dataloaders["test"])
                print_metrics(metrics_val["overall"], epoch, "val")
            else:
                metrics_val = _eval_epoch(dataloaders["test"])
                overall_metrics = metrics_val.pop("overall")
                print("")
                for category, metrics in metrics_val.items():
                    print_metrics(metrics, epoch, category)
                print_metrics(overall_metrics, epoch, "overall")

        if type(model) == LinearProbingNet and epoch == num_epochs:
            print("Against full resolution annotations")
            metrics_val = _eval_epoch(dataloaders["test"], full_resolution=True)
            overall_metrics = metrics_val.pop("overall")
            print("")
            for category, metrics in metrics_val.items():
                print_metrics(metrics, epoch, category)
            print_metrics(overall_metrics, epoch, "overall")


def visualize_model(
    model: nn.Module, dataset: DavisDataset, index: int, device: torch.device
) -> None:
    if index > len(dataset):
        index = len(dataset)
    data = dataset[index]

    feature_maps = data["feature_map"].to(device)
    image = data["image"].to(device)
    kwargs = {"guide": image[None]}
    segmentation = binary_output_to_labels(
        model(feature_maps[None].permute(0, 3, 1, 2), **kwargs)
    )

    H_seg, W_seg = segmentation.shape[-2], segmentation.shape[-1]
    H, W = data["image"].shape[-2], data["image"].shape[-1]
    scale_factor = H // H_seg
    upsample = nn.modules.upsampling.Upsample(scale_factor=scale_factor, mode="nearest")

    fig, axs = plt.subplots(1, 2, dpi=dpi, figsize=(20, 5))

    axs[0].imshow(
        data["image"].mul(255).permute(1, 2, 0).byte().cpu().numpy(),
        cmap="gray",
        interpolation="none",
    )
    axs[0].imshow(
        upsample(segmentation)[0, 0].mul(255).byte().numpy(),
        cmap="jet",
        alpha=0.5,
        interpolation="none",
    )
    axs[0].set_title("Image", fontsize=20)
    axs[0].axis("off")

    axs[1].imshow(
        data["image"].mul(255).permute(1, 2, 0).byte().cpu().numpy(),
        cmap="gray",
        interpolation="none",
    )
    axs[1].imshow(
        data["annotations"].mul(255).byte().numpy(),
        cmap="jet",
        alpha=0.5,
        interpolation="none",
    )
    axs[1].set_title("With Annotations", fontsize=20)
    axs[1].axis("off")

    plt.show()
