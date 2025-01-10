import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from exercise_code.data.utils import feature_map_pca, load_annotation, load_feature_map, load_img, save_annotation

dpi = 96


class DavisDataset(Dataset):
    def __init__(self, path: Path, patch_size: int) -> None:
        self.dataset_path = path
        self.patch_size = patch_size
        self.image_dir = "images"
        self.annotation_dir = "annotations"
        self.annotation_coarse_dir = "annotations_coarse"
        self.feature_maps_dir = "embeddings"
        self.image_names: List[Tuple[str, str]] = []
        self.category_names = []

        for category_dir in sorted(self.dataset_path.iterdir()):
            if not (
                category_dir.is_dir()
                and category_dir.joinpath(self.image_dir).is_dir()
                and category_dir.joinpath(self.feature_maps_dir).is_dir()
                and category_dir.joinpath(self.annotation_dir).is_dir()
                and category_dir.joinpath(self.annotation_coarse_dir).is_dir()
            ):
                continue
            category_name = category_dir.parts[-1]
            for image in sorted(category_dir.joinpath(self.image_dir).iterdir()):
                if not image.is_file():
                    continue
                name = image.stem
                if (
                    category_dir.joinpath(self.feature_maps_dir).joinpath(f"{name}.pt").is_file()
                    and category_dir.joinpath(self.annotation_dir).joinpath(f"{name}.png").is_file()
                    and category_dir.joinpath(self.annotation_coarse_dir).joinpath(f"{name}.png").is_file()
                ):
                    self.image_names.append((name, category_name))
                    if category_name not in self.category_names:
                        self.category_names.append(category_name)

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index) -> Dict[str, Any]:
        name, category_name = self.image_names[index]

        img = load_img(self.dataset_path.joinpath(category_name, self.image_dir, f"{name}.jpg"))
        feature_map = load_feature_map(self.dataset_path.joinpath(category_name, self.feature_maps_dir, f"{name}.pt"))
        annotations = load_annotation(self.dataset_path.joinpath(category_name, self.annotation_dir, f"{name}.png"))
        annotations_coarse = load_annotation(
            self.dataset_path.joinpath(category_name, self.annotation_coarse_dir, f"{name}.png")
        )

        img = img[
            ...,
            : math.floor(img.shape[-2] / self.patch_size) * self.patch_size,
            : math.floor(img.shape[-1] / self.patch_size) * self.patch_size,
        ]
        annotations = annotations[
            ...,
            : math.floor(annotations.shape[-2] / self.patch_size) * self.patch_size,
            : math.floor(annotations.shape[-1] / self.patch_size) * self.patch_size,
        ]

        return {
            "image": img,
            "feature_map": feature_map,
            "annotations": annotations,
            "annotations_coarse": annotations_coarse,
            "category": category_name,
        }


def create_embeddings(path: Path, dino_model, device: torch.device) -> None:
    image_dir = "images"
    feature_map_dir = "embeddings"
    P = dino_model.patch_embed.patch_size
    num_heads = dino_model.blocks[0].attn.num_heads
    feat_out = {}

    def hook_fn_forward_qkv(module, input, output):
        feat_out["qkv"] = output

    dino_model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

    for category_dir in sorted(path.iterdir()):
        if not (category_dir.is_dir() and category_dir.joinpath(image_dir).is_dir()):
            continue
        category_dir.joinpath(feature_map_dir).mkdir(parents=True, exist_ok=True)

        image_paths = [
            image_path
            for image_path in sorted(category_dir.joinpath(image_dir).iterdir())
            if image_path.suffix == ".jpg"
        ]

        for image_path in tqdm(image_paths):
            image = load_img(image_path)[None].to(device)

            image = image[
                ...,
                : math.floor(image.shape[-2] / dino_model.patch_embed.patch_size) * dino_model.patch_embed.patch_size,
                : math.floor(image.shape[-1] / dino_model.patch_embed.patch_size) * dino_model.patch_embed.patch_size,
            ]

            f_map = dino_model.get_intermediate_layers(image, n=1)[0]

            B, C, H, W = image.shape
            H_patch, W_patch = H // P, W // P
            H_pad, W_pad = H_patch * P, W_patch * P
            T = H_patch * W_patch + 1
            output_qkv = feat_out["qkv"].reshape(B, T, 3, num_heads, -1 // num_heads).permute(2, 0, 3, 1, 4)
            # output_dict['q'] = output_qkv[0].transpose(1, 2).reshape(B, T, -1)[:, 1:, :]
            keys = output_qkv[1].transpose(1, 2).reshape(B, T, -1)[:, 1:, :]
            print(keys.shape)
            print(f_map.shape)
            print(keys)
            print(f_map)

            h, w = int(image.shape[-2] / dino_model.patch_embed.patch_size), int(
                image.shape[-1] / dino_model.patch_embed.patch_size
            )
            dim = f_map.shape[-1]
            f_map = f_map[0][1:].reshape(h, w, dim)

            f_map = f_map.to(torch.float16)
            torch.save(f_map, category_dir.joinpath(feature_map_dir, f"{image_path.stem}.pt"))


def max_pool_downsampling(patch_size) -> Callable[[torch.Tensor], torch.Tensor]:
    downsampling = nn.MaxPool2d(patch_size, patch_size)

    def _downsample(annotation: torch.Tensor) -> torch.Tensor:
        return downsampling(annotation)

    return _downsample


def avg_pool_downsampling(patch_size) -> Callable[[torch.Tensor], torch.Tensor]:
    downsampling = nn.AvgPool2d(patch_size, patch_size)

    def _downsample(annotation: torch.Tensor) -> torch.Tensor:
        annotation_downsampled = downsampling(annotation)
        annotation_downsampled[annotation_downsampled < 0.5] = 0
        annotation_downsampled[annotation_downsampled >= 0.5] = 1
        return annotation_downsampled

    return _downsample


def downsample_annotations(path: Path, patch_size: int) -> None:
    annotations_dir = "annotations"
    annotations_coarse_dir = "annotations_coarse"
    for category_dir in sorted(path.iterdir()):
        if not (category_dir.is_dir() and category_dir.joinpath(annotations_dir).is_dir()):
            continue
        category_dir.joinpath(annotations_coarse_dir).mkdir(parents=True, exist_ok=True)

        annotation_paths = [
            annotation_path
            for annotation_path in sorted(category_dir.joinpath(annotations_dir).iterdir())
            if annotation_path.suffix == ".png"
        ]

        downsampling = max_pool_downsampling(patch_size)

        for annotation_path in tqdm(annotation_paths):
            annotation = load_annotation(annotation_path)

            annotation = annotation[
                ...,
                : math.floor(annotation.shape[-2] / patch_size) * patch_size,
                : math.floor(annotation.shape[-1] / patch_size) * patch_size,
            ]

            annotation_coarse = downsampling(annotation[None])[0]

            save_annotation(category_dir.joinpath(annotations_coarse_dir, annotation_path.parts[-1]), annotation_coarse)


def load_davis_dataset(path: Path, patch_size: int) -> Tuple[DavisDataset, DavisDataset]:
    return DavisDataset(path.joinpath("train"), patch_size), DavisDataset(path.joinpath("test"), patch_size)


# ==================================================== Visualization ===================================================
def visualize_davis(dataset: DavisDataset, index: int) -> None:
    if index > len(dataset):
        index = len(dataset)
    data = dataset[index]

    H_feat, W_feat = data["feature_map"].shape[0], data["feature_map"].shape[1]
    H, W = data["image"].shape[-2], data["image"].shape[-1]
    scale_factor = H // H_feat
    upsample = nn.modules.upsampling.Upsample(scale_factor=scale_factor, mode="nearest")

    pca_map = feature_map_pca(data["feature_map"].flatten(0, 1)).reshape(H_feat, W_feat, 3)

    fig, axs = plt.subplots(1, 4, dpi=dpi, figsize=(20, 5))

    axs[0].imshow(data["image"].mul(255).permute(1, 2, 0).byte().cpu().numpy(), cmap="gray", interpolation="none")
    axs[0].set_title("Image", fontsize=20)
    axs[0].axis("off")

    axs[1].imshow(data["image"].mul(255).permute(1, 2, 0).byte().cpu().numpy(), cmap="gray", interpolation="none")
    axs[1].imshow(data["annotations"].mul(255).byte().numpy(), cmap="jet", alpha=0.5, interpolation="none")
    axs[1].set_title("With Annotations", fontsize=20)
    axs[1].axis("off")

    axs[2].imshow(data["image"].mul(255).permute(1, 2, 0).byte().cpu().numpy(), cmap="gray", interpolation="none")
    axs[2].imshow(
        upsample(data["annotations_coarse"][None, None])[0, 0].mul(255).byte().numpy(),
        cmap="jet",
        alpha=0.5,
        interpolation="none",
    )
    axs[2].set_title("With coarse annotations", fontsize=20)
    axs[2].axis("off")

    axs[3].imshow(
        upsample(pca_map.permute(2, 0, 1)[None])[0].permute(1, 2, 0).mul(255).byte().cpu().numpy(),
        cmap="gray",
        interpolation="none",
    )
    axs[3].set_title("PCA visualization of embeddings", fontsize=20)
    axs[3].axis("off")

    plt.show()
