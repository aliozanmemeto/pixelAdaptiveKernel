import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from exercise_code.data.obj_segmentation.pac_backend import PacConv2d


class PACNet(nn.Module):
    def __init__(self, patch_size: int = 16) -> None:
        super().__init__()

        linear_out_layers = 16
        self.linear = nn.Conv2d(384, linear_out_layers, kernel_size=1, bias=True)

        if patch_size == 16:
            channel_dim = [linear_out_layers, 4, 4, 4, 1]
        else:
            channel_dim = [linear_out_layers, 4, 4, 1]

        self.pac_layers = nn.ModuleList()
        for i in range(1, len(channel_dim)):
            self.pac_layers.append(
                PacConv2d(
                    channel_dim[i - 1],
                    channel_dim[i],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        guide = guide[
            ...,
            : math.floor(guide.shape[-1] / 16) * 16,
            : math.floor(guide.shape[-1] / 16) * 16,
        ]
        x = self.linear(x)
        x = F.leaky_relu(x)

        for i in range(len(self.pac_layers)):
            curr_guide = F.interpolate(
                guide,
                scale_factor=1 / 2 ** (len(self.pac_layers) - (i + 1)),
                align_corners=False,
                mode="bilinear",
            )
            # print(curr_guide.shape)

            x = F.interpolate(x, scale_factor=2)
            x = self.pac_layers[i](x, curr_guide)

            if i < len(self.pac_layers) - 1:
                x = F.leaky_relu(x)

        return F.sigmoid(x)
