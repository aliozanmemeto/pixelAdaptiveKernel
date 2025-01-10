import json
from pathlib import Path
from typing import Any, Dict, Optional
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset
from panopticapi.utils import IdGenerator

from exercise_code.data.utils import load_img, load_semantic_annotation, load_instance_annotation, combine_annotation

dpi = 96


class CityscapeDataset(Dataset):
    def __init__(self, dataset_path: Path) -> None:
        super().__init__()

        self.image_dir = dataset_path.joinpath("images")
        self.annotation_dir = dataset_path.joinpath("annotations")

        self.images = []
        for img in sorted(self.image_dir.iterdir()):
            name = img.stem[:-12]
            if (
                self.annotation_dir.joinpath(f"{name}_gtFine_color.png").is_file()
                and self.annotation_dir.joinpath(f"{name}_gtFine_instanceIds.png").is_file()
            ):
                self.images.append(name)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        name = self.images[index]

        image = load_img(self.image_dir.joinpath(f"{name}_leftImg8bit.png"))
        semantic_annotation = load_semantic_annotation(self.annotation_dir.joinpath(f"{name}_gtFine_color.png"))
        instance_annotation = load_instance_annotation(self.annotation_dir.joinpath(f"{name}_gtFine_instanceIds.png"))

        panoptic_annotation = combine_annotation(semantic_annotation, instance_annotation)

        return {
            "image": image,
            "semantic": semantic_annotation,
            "instance": instance_annotation,
            "panoptic": panoptic_annotation,
        }


class CityscapeDatasetAlt(Dataset):
    def __init__(self, dataset_dir: Path) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        self.image_dir = dataset_dir.joinpath("images")
        self.annotation_dir = dataset_dir.joinpath("annotations")

        self.name = self.dataset_dir.parts[-1]

        with open(self.dataset_dir.joinpath(f"annotation_info.json")) as json_file:
            data = json.load(json_file)

        self.images = [
            image for image in data["images"] if self.image_dir.joinpath(f"{image['id']}_leftImg8bit.png").is_file()
        ]
        self.annotations = [
            annotation
            for annotation in data["annotations"]
            if self.image_dir.joinpath(f"{annotation['image_id']}_leftImg8bit.png").is_file()
        ]
        self.categories = data["categories"]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index) -> Dict[str, Any]:
        name = self.images[index]["id"]

        # image
        image = load_img(self.image_dir.joinpath(f"{name}_leftImg8bit.png"))

        # annotation
        file_name = ""
        idx = 0
        for idx, annotation in enumerate(self.annotations):
            if annotation["image_id"] == name:
                file_name = annotation["file_name"]
                break

        panoptic_file = self.annotation_dir.joinpath(file_name)

        panoptic_annotation = load_semantic_annotation(panoptic_file).long()
        ids = panoptic_annotation[0] + (256 * panoptic_annotation[1]) + (256 * 256 * panoptic_annotation[2])

        return {f"image": image, f"panoptic_ids": ids, "annotation_id": idx}


def panoptic_to_color(ids: torch.Tensor, segment_infos, categories):
    panoptic_segmentation = torch.zeros_like(ids)[..., None].repeat(1, 1, 3)
    unique_ids = ids.unique()

    color_generator = IdGenerator(categories)

    for id in unique_ids:
        mask = ids == id
        segment_info = segment_infos[0]
        for info in segment_infos:
            if id == info["id"]:
                segment_info = info
        color = color_generator.get_color(segment_info["category_id"])
        panoptic_segmentation[mask] = torch.tensor(color)

    return panoptic_segmentation


def visualize_panoptic_segmentation(
    img: torch.Tensor, gt_panoptic: Dict[str, Any], predicted_panoptic: Optional[Dict[str, Any]] = None
):
    num_imgs = 2
    if predicted_panoptic is not None:
        num_imgs = 3

    fig, axs = plt.subplots(1, num_imgs, dpi=dpi, figsize=(5 * num_imgs, 5))

    axs[0].imshow(img.mul(255).permute(1, 2, 0).byte().cpu().numpy(), cmap="gray", interpolation="none")
    axs[0].set_title("Image", fontsize=20)
    axs[0].axis("off")

    axs[1].imshow(
        panoptic_to_color(gt_panoptic["ids"], gt_panoptic["segments_info"], gt_panoptic["categories"])
        .byte()
        .cpu()
        .numpy(),
        cmap="gray",
        interpolation="none",
    )
    axs[1].set_title("GT Panoptic Seg", fontsize=20)
    axs[1].axis("off")

    if predicted_panoptic is not None:
        axs[2].imshow(
            panoptic_to_color(
                predicted_panoptic["ids"], predicted_panoptic["segments_info"], predicted_panoptic["categories"]
            )
            .byte()
            .cpu()
            .numpy(),
            cmap="gray",
            interpolation="none",
        )
        axs[2].set_title("Predicted Panoptics", fontsize=20)
        axs[2].axis("off")

    plt.show()
