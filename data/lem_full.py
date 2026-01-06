import lightning as L
import torch
import numpy as np
import json
import tifffile as tiff
from pathlib import Path
from einops import rearrange
from typing import Any, Callable
from scipy.ndimage import generic_filter
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import Compose

from .transforms import (
    ToTensor,
    SampleTimestamps,
    Crop
)


class LEM_Full(Dataset):
    def __init__(
            self,
            data_dir: str | Path,
            patch_size: int,
            overlap: float = 0.5,
            normalize: bool = True,
            transform: Callable[[dict], Any] | None = None
        ) -> None:
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.transform = transform
        
        self.city_id = "2919553"
        self.means_stds = self._load_means_stds()
        self.full_img = self._load_full_img()

        self.patch_size = patch_size
        self.stride = int(patch_size * (1 - overlap))

        self.H = self.full_img["data"].shape[-2]
        self.W = self.full_img["data"].shape[-1]

        self.coords = self._compute_coords()
    
    def __len__(self) -> int:
        return len(self.coords)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        top, left = self.coords[idx]
        data, target = self.full_img["data"], self.full_img["target"]

        data = data[..., top:top+self.patch_size, left:left+self.patch_size]
        data = np.nan_to_num(data).astype(np.float32)

        if self.normalize:
            data = self._normalize_data(data)
                
        data = self._sample_channels(data)

        target = target[top:top+self.patch_size, left:left+self.patch_size]
        target = self._process_labels(target)

        sample = {
            "data": data,
            "target": target,
            "coords": torch.tensor([top, left], dtype=torch.int32)
        }

        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _load_means_stds(self) -> dict[str, dict[str, float]]:
        mean_std_file_path = self.data_dir / "means_stds.json"
        
        if not mean_std_file_path.exists():
            raise FileNotFoundError(f"Mean and standard deviation file not found at {mean_std_file_path}")
        
        with open(mean_std_file_path, "r") as f:
            norm_dict = json.load(f)
        
        return norm_dict

    def _load_full_img(self) -> dict[str, Any]:
        data_ts1 = tiff.imread(self.data_dir / "raw" / "2919553_April_2023_s2.tif")
        data_ts2 = tiff.imread(self.data_dir / "raw" / "2919553_May_2023_s2.tif")
        data_ts3 = tiff.imread(self.data_dir / "raw" / "2919553_June_2023_s2.tif")

        data = np.stack([data_ts1, data_ts2, data_ts3], axis=0)
        data = rearrange(data, "t h w c -> t c h w")
        
        target = tiff.imread(self.data_dir / "raw" / "lem_reference_instance.tif")
        
        return {
            "data": data,
            "target": target
        }
    
    def _sample_channels(self, data: np.ndarray) -> np.ndarray:
        # Excluding 60 m channels (0 and 9)
        valid_channels = list(set(range(12)) - set([0, 9]))
        
        return data[:, valid_channels]
        
    def _normalize_data(self, data) -> np.ndarray:
        mean = np.array(self.means_stds[self.city_id]["mean"]).astype(np.float32)
        std = np.array(self.means_stds[self.city_id]["std"]).astype(np.float32)
        
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        
        # Reverting original normalization
        data = data * 10_000

        return (data - mean) / std
    
    def _binarize_labels(self, target: np.ndarray) -> np.ndarray:
        return np.where(target > 0, 1, 0).astype(np.float32)

    def _process_labels(self, target: np.ndarray) -> np.ndarray:
        def compute_margins(target):
            return len(np.unique(target)) > 1
        
        # Binarizing mask for background vs crop
        bin_mask = (target > 0).astype(int)
        bin_mask[bin_mask > 0] = 1
        
        mrg_size = 5

        margin = generic_filter(target, compute_margins, size=[mrg_size, mrg_size])
        bin_mask[margin > 0] = 2
        
        return bin_mask

    def _compute_coords(self) -> list[tuple[int, int]]:
        data = self.full_img["data"]
        coords = []

        for top in range(0, self.H - self.patch_size + 1, self.stride):
            for left in range(0, self.W - self.patch_size + 1, self.stride):
                patch = data[..., top:top+self.patch_size, left:left+self.patch_size]

                # Skipping all-zero and all-NaN patches
                if np.all(patch == 0) or np.isnan(patch).all():
                    continue

                coords.append((top, left))

        return coords


class LEM_Full_Module(L.LightningDataModule):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()

        self.patch_size = config.model.patch_size
        self.num_workers = config.dataset.num_workers
        self.batch_size = config.dataset.batch_size
        self.num_timestamps = config.model.num_frames
        self.img_size = config.model.img_size
        self.data_dir = Path(config.dataset.data_dir)

        self.transform = Compose([
            SampleTimestamps(num_timestamps=self.num_timestamps, sample_type="first"),
            Crop(size=(self.img_size, self.img_size), crop_type="center"),
            ToTensor()
        ])

    def setup(self, stage: str | None = None) -> None:
        """
        Setup the dataset for training, validation, and testing.
        """
        if stage == "test" or stage is None:
            self.test_dataset = LEM_Full(self.data_dir, patch_size=self.patch_size, transform=self.transform)

    def test_dataloader(self) -> Callable[[dict], DataLoader]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        )