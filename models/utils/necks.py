import torch
import torch.nn as nn
from einops import rearrange


class SampleFeatures(nn.Module):
    """
    Samples tensors from a list of features
    """
    def __init__(self, indices: list[int]) -> None:
        super().__init__()
        self.indices = indices

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(self.indices) == len(features):
            return features

        return [features[i] for i in self.indices]


class ReshapeFeatures(nn.Module):
    """
    Reshapes encoder features for the decoder

    input: [B, H*W, D]
    output: [B, D, H, W]
    """
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        reshaped_features = []

        for i, feature in enumerate(features):
            if len(feature.shape) == 4:
                if feature.shape[-1] != feature.shape[-2]:
                    feature = rearrange(feature, "b h w d -> b d h w")
            else:
                _, N, _ = feature.shape
                feature = feature[:, 1:, :] # Remove class token
                H = W = int(N**0.5)
                feature = rearrange(feature, "b (h w) d -> b d h w", h=H, w=W)

            reshaped_features.append(feature)
            
        return reshaped_features