import lightning as L
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Any, Callable
from scipy.ndimage import generic_filter
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import Compose

from .transforms import (
    ToTensor,
    SampleTimestamps,
    RandomFlip,
    RandomRotate,
    RandomContrast,
    Crop
)


class LEM(Dataset):
    def __init__(
            self,
            data_dir: str | Path,
            split: str,
            normalize: bool = True,
            with_datetime: bool = True,
            transform: Callable[[dict], Any] | None = None
        ) -> None:
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.normalize = normalize
        self.with_datetime = with_datetime
        self.transform = transform
        
        self.metadata = self._load_metadata()
        self.means_stds = self._load_means_stds()
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        file_path = self.data_dir / self.metadata.iloc[idx]["file_path"]
        
        with open(file_path, "rb") as handle:
            pickle_data = pickle.load(handle, encoding="latin1")

        data = np.nan_to_num(pickle_data["img"]).astype(np.float32)
        stem = str(self.metadata.iloc[idx]["stem"])
        
        if self.normalize:
            data = self._normalize_data(data, stem)

        data = self._sample_channels(data)
        
        sample = {
            "data": data,
            "stem": stem
        }
        
        target = pickle_data["labels"]
        sample["target"] = self._process_labels(target)
        
        if self.with_datetime:
            sample["dates"] = np.array(pickle_data["doy"]).astype(np.float32)

        if self.transform:
            sample = self.transform(sample)
        
        return sample
        
    def _load_metadata(self) -> pd.DataFrame:
        assert self.split in ["train", "val", "test"], f"Invalid split {self.split}. \
                                                        Choose from ['train', 'val', 'test']"

        metadata_file_path = self.data_dir / f"{self.split}_metadata.csv"
        
        if not metadata_file_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_file_path}")
        
        metadata_df = pd.read_csv(metadata_file_path)
        metadata_df = metadata_df.reset_index(drop=True)
        
        return metadata_df
    
    def _load_means_stds(self) -> dict[str, dict[str, float]]:
        mean_std_file_path = self.data_dir / "means_stds.json"
        
        if not mean_std_file_path.exists():
            raise FileNotFoundError(f"Mean and standard deviation file not found at {mean_std_file_path}")
        
        with open(mean_std_file_path, "r") as f:
            norm_dict = json.load(f)
        
        return norm_dict
    
    def _sample_channels(self, data: np.ndarray) -> np.ndarray:
        # Excluding 60 m channels (0 and 9)
        valid_channels = list(set(range(12)) - set([0, 9]))
        
        return data[:, valid_channels]
        
    def _normalize_data(self, data: np.ndarray, stem: str) -> np.ndarray:
        stem = stem.split("_")[0]
        
        mean = np.array(self.means_stds[stem]["mean"]).astype(np.float32)
        std = np.array(self.means_stds[stem]["std"]).astype(np.float32)
        
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


class LEM_Module(L.LightningDataModule):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()

        self.validate = config.dataset.validate
        self.batch_size = config.dataset.batch_size
        self.num_workers = config.dataset.num_workers
        self.num_timestamps = config.model.num_frames
        self.img_size = config.model.img_size
        self.data_dir = Path(config.dataset.data_dir)

        self.transform_train = Compose([
            SampleTimestamps(num_timestamps=self.num_timestamps, sample_type="random"),
            Crop(size=(self.img_size, self.img_size), crop_type="random"),
            RandomFlip(prob=0.5, orientation="hor"),
            RandomFlip(prob=0.5, orientation="ver"),
            RandomRotate(),
            RandomContrast(),
            ToTensor()
        ])
        self.transform_test = Compose([
            SampleTimestamps(num_timestamps=self.num_timestamps, sample_type="first"),
            Crop(size=(self.img_size, self.img_size), crop_type="center"),
            ToTensor()
        ])

    def setup(self, stage: str | None = None) -> None:
        """
        Setup the dataset for training, validation, and testing.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = LEM(self.data_dir, split="train", transform=self.transform_train)
            
            if self.validate:
                self.val_dataset = LEM(self.data_dir, split="val", transform=self.transform_test)
        if stage == "test" or stage is None:
            self.test_dataset = LEM(self.data_dir, split="test", transform=self.transform_test)

    def train_dataloader(self) -> Callable[[dict], DataLoader]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True
        )
    
    def val_dataloader(self) -> Callable[[dict], DataLoader] | None:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        ) if self.validate else None

    def test_dataloader(self) -> Callable[[dict], DataLoader]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        )