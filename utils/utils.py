import os
import torch
import torchvision
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.utils import make_grid
from typing import Callable, Any
from pathlib import Path
from tqdm import tqdm


def get_data_sample(dataset: Dataset, indices: list[int] | int) -> DataLoader:
    """
    Generate a dataloader from a sample of the data. The sample is random unless indices are specified.
    """
    if isinstance(indices, int):
        indices = random.sample(range(len(dataset)), k=indices)
        
    subset = Subset(dataset, indices)
    subset_loader = DataLoader(subset, batch_size=16)
    
    return subset_loader


def store_preds_as_images(batch: dict[str, torch.Tensor], save_dir: str | Path) -> None:
    """
    Given a batch of time series, store each time series as a grid PNG image
    """
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
        images = []
        
        for t in range(time_series.shape[1]):
            image = time_series[:, t, :, :].detach().cpu().numpy()
            
            rgb_image = image[[2, 1, 0], :, :] # Indices for B04 (red), B03 (green), B02 (blue)
            rgb_image = np.stack([normalize_image(rgb_image[i]) for i in range(3)], axis=0)
            
            images.append(rgb_image)

        return images

    print("Storing batch predictions...")
    save_dir = Path(save_dir) / "predictions"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    inputs, preds = batch["inputs"], batch["preds"]
    
    for idx, (series, pred) in tqdm(enumerate(zip(inputs, preds)), total=len(inputs)):        
        series = time_series_to_rgb(series)
        pred = time_series_to_rgb(pred)
        
        combined = np.concatenate([series, pred], axis=0)
        grid = make_grid(torch.tensor(combined), nrow=len(series), padding=5, pad_value=255)

        grid_image = torchvision.transforms.ToPILImage()(grid)
        grid_image.save(save_dir / f"{idx}.png")


def get_random_embedding(patch_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Return a random embedding of the given patch embeddings that is not the cls token.
    """
    assert len(patch_embeddings.shape) == 3, "Expected patch_embeddings to be of shape (batch_size, num_patches, embedding_dim)"
    
    num_patches = patch_embeddings.shape[1]
    patch_index = torch.randint(1, num_patches, (1,)).item()
    
    return patch_embeddings[:, patch_index]
