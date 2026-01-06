import os
import torch
import torchvision
import random
import numpy as np
import lightning as L
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.utils import make_grid
from typing import Callable, Any
from pathlib import Path
from tqdm import tqdm
from einops import rearrange


def get_data_sample(dataset: Dataset, indices: list[int] | int) -> DataLoader:
    """
    Generate a dataloader from a sample of the data. The sample is random unless indices are specified.
    """
    if isinstance(indices, int):
        indices = random.sample(range(len(dataset)), k=indices)
        
    subset = Subset(dataset, indices)
    subset_loader = DataLoader(subset, batch_size=16)
    
    return subset_loader


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to 0-255 for visualization.
    """
    min_val = np.amin(image, axis=(-2, -1), keepdims=True)
    max_val = np.amax(image, axis=(-2, -1), keepdims=True)
    
    image = np.clip((image - min_val) / (max_val - min_val + 1e-5), 0, 1)
    
    return (image * 255).astype(np.uint8)


def time_series_to_rgb(time_series: torch.Tensor) -> list[np.ndarray]:
    """
    Convert a time series to RGB images.
    """
    if len(time_series.shape) == 3:
        # Adding a time dimension if images are 2D
        time_series = time_series.unsqueeze(1)

    images = []
    
    for t in range(time_series.shape[1]):
        image = time_series[:, t, :, :].detach().cpu().numpy()
        
        rgb_image = image[[2, 1, 0], :, :] # Indices for B04 (red), B03 (green), B02 (blue)
        rgb_image = np.stack([normalize_image(rgb_image[i]) for i in range(3)], axis=0)
        rgb_image = rearrange(rgb_image, "c h w -> h w c")
        
        images.append(rgb_image)

    return images


def store_preds_as_images(batch: dict[str, torch.Tensor], save_dir: str | Path) -> None:
    """
    Given a batch of time series, store each time series as a grid PNG image
    """
    save_dir = Path(save_dir) / "predictions"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    inputs, preds = batch["inputs"], batch["preds"]
    
    for idx, (image, pred) in tqdm(enumerate(zip(inputs, preds)), total=len(inputs)):
        image = time_series_to_rgb(image)
        pred = time_series_to_rgb(pred)
        
        combined = np.concatenate([image, pred], axis=0)
        grid = make_grid(torch.tensor(combined), nrow=len(image), padding=5, pad_value=255)

        grid_image = torchvision.transforms.ToPILImage()(grid)
        grid_image.save(save_dir / f"{idx}.png")
        

def get_preds_from_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    if len(logits.shape) < 4:
        logits = logits.unsqueeze(0) # [B, C, H, W]

    if num_classes == 1:
        # Binary case: sigmoid + threshold
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
    else:
        # Multiclass case: argmax
        preds = torch.argmax(logits, dim=1)

    return preds


def load_from_checkpoint(
        config: dict[str, Any],
        model: L.LightningModule,
        model_name: Callable[[dict], L.LightningModule]
    ) -> L.LightningModule:
    if config.checkpoint.ckpt_path is not None:
        # Loads ALL model weights
        model = model_name.load_from_checkpoint(
            checkpoint_path=config.checkpoint.ckpt_path,
            weights_only=False,
            config=config
        )
    elif config.checkpoint.pretrain_weights is not None:
        # Loads only encoder weights for segmentation
        print("Loading pretrained encoder weights...")

        checkpoint = torch.load(config.checkpoint.pretrain_weights, weights_only=False, map_location="cpu")
        state_dict = checkpoint["state_dict"]

        for key in list(state_dict.keys()):
            new_key = key.replace("backbone.", "")
            state_dict[new_key] = state_dict.pop(key)
        
        model.backbone.load_state_dict(state_dict, strict=True)
    
    return model


def get_random_embedding(patch_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Return a random embedding of the given patch embeddings that is not the cls token.
    """
    if len(patch_embeddings.shape) == 4:
        patch_embeddings = rearrange(patch_embeddings, "b h w d -> b (h w) d")

    assert len(patch_embeddings.shape) == 3, "Expected patch_embeddings to be of shape (batch_size, num_patches, embedding_dim)"
    
    num_patches = patch_embeddings.shape[1]
    patch_index = torch.randint(1, num_patches, (1,)).item()
    
    return patch_embeddings[:, patch_index]
