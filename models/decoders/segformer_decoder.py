import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.base import SegmentationHead


class MLP(nn.Module):
    def __init__(self, skip_channels: int, decoder_dim: int):
        super().__init__()

        self.linear = nn.Linear(skip_channels, decoder_dim)

    def forward(self, x: torch.Tensor):
        batch, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.linear(x)
        x = x.transpose(1, 2).reshape(batch, -1, height, width)
        return x


class SegformerDecoder(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        embed_dims: list[int] = [768, 768, 768, 768],
        decoder_dim: int = 256,
        num_classes: int = 1000
    ):
        super().__init__()
        self.img_size = img_size

        self.mlp_stage = nn.ModuleList(
            [MLP(channel, decoder_dim) for channel in embed_dims]
        )
        self.fuse_stage = md.Conv2dReLU(
            in_channels=(len(embed_dims) * decoder_dim),
            out_channels=decoder_dim,
            kernel_size=1,
            use_norm="batchnorm",
        )
        self.head = SegmentationHead(
            in_channels=decoder_dim,
            out_channels=num_classes,
            kernel_size=1,
            upsampling=4
        )

    def forward(self, x: torch.Tensor, features: list[torch.Tensor]) -> torch.Tensor:
        # Resize all features to the size of the largest feature
        target_size = self.img_size // 4
        features = features[::-1]  # reverse channels to start from head of encoder
        
        resized_features = []
        for i, mlp_layer in enumerate(self.mlp_stage):
            feature = mlp_layer(features[i])
            resized_feature = F.interpolate(
                feature, size=target_size, mode="bilinear", align_corners=False
            )
            resized_features.append(resized_feature)

        output = self.fuse_stage(torch.cat(resized_features, dim=1))
        pred = self.head(output)

        return pred


if __name__ == "__main__":
    decoder = SegformerDecoder(
        num_classes=3
    )
    
    features = [torch.randn(16, 145, 768) for _ in range(4)]
    # features = [
    #     torch.randn(16, 768, 24, 24),
    #     torch.randn(16, 768, 12, 12),
    #     torch.randn(16, 768, 6, 6),
    #     torch.randn(16, 768, 3, 3)
    # ]
    pred = decoder(features)
    
    print(pred.shape)
