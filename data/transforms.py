import torch
import os
import numpy as np
import pandas as pd
from einops import rearrange
from pathlib import Path


class ToTensor:
    def __init__(self, with_datetime: bool = True) -> None:
        self.with_datetime = with_datetime
        
    def __call__(self, sample: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        """
        Convert the input sample to a tensor.
        """
        data = sample["data"]
        assert len(data.shape) == 4, "Expected a 4D array"
        
        data = rearrange(data, "t c h w -> c t h w")
        sample["data"] = torch.from_numpy(data.copy())
        
        if "target" in sample.keys():
            target = sample["target"]
            assert len(target.shape) == 3, "Expected a 3D array for the target"

            sample["target"] = torch.from_numpy(target.copy())
        
        if self.with_datetime:
            dates = sample["dates"]
            sample["dates"] = torch.from_numpy(dates.copy())
        
        return sample


class SampleRandomTimestamps:
    def __init__(self, num_timestamps: int = 3, with_datetime: bool = True) -> None:
        self.num_timestamps = num_timestamps
        self.with_datetime = with_datetime

    def __call__(self, sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Sample random timestamps from the input data.
        """
        data = sample["data"]

        if data.shape[0] < self.num_timestamps:
            raise ValueError(f"Number of timestamps in data is less than {num_timestamps}.")

        timestamps = np.sort(np.random.choice(data.shape[0], size=self.num_timestamps, replace=False))
        sample["data"] = data[timestamps]
        
        if self.with_datetime:
            dates = sample["dates"]
            sample["dates"] = dates[timestamps]

        return sample


class RandomFlip:
    def __init__(self, prob: float = 0.5, orientation: str = "hor") -> None:
        self.prob = prob
        self.orientation = orientation

    def __call__(self, sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Randomly flip the data horizontally with probability prob.
        """
        if self.orientation == "hor":
            axis = -1
        elif self.orientation == "ver":
            axis = -2
        else:
            raise ValueError("Orientation must be either 'hor' or 'ver'.")
        
        if np.random.rand() < self.prob:
            sample["data"] = np.flip(sample["data"], axis=axis)
            
            if "target" in sample.keys():
                sample["target"] = np.flip(sample["target"], axis=axis)
        
        return sample


class RandomRotate:
    def __call__(self, sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Rotate images and labels with a random degree in {0, 90, 180, 270}.
        """
        num_rotations = np.random.randint(4)
        
        if num_rotations > 0:
            sample["data"] = np.rot90(sample["data"], k=num_rotations, axes=(-2, -1))
            
            if "target" in sample.keys():
                sample["target"] = np.rot90(sample["target"], k=num_rotations, axes=(-2, -1))
        
        return sample


class Crop:
    def __init__(self, size: tuple[int, int], crop_type: str = "random") -> None:
        self.size = size
        self.crop_type = crop_type
    
    def __call__(self, sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Crop a time series to a new size.
        """
        data = sample["data"]
        assert len(data.shape) == 4, "Expected a 4D array for the data"
        
        _, _, h, w = data.shape
        crop_h, crop_w = self.size
        
        if h < crop_h or w < crop_w:
            raise ValueError("Crop size must be smaller than the image size.")
        
        if self.crop_type == "random":
            top = np.random.randint(0, h - crop_h + 1)
            left = np.random.randint(0, w - crop_w + 1)
        elif self.crop_type == "center":
            top = (h - crop_h) // 2
            left = (w - crop_w) // 2
        else:
            raise ValueError("Crop type must be either 'random' or 'center'.") 
        
        sample["data"] = data[:, :, top:top+crop_h, left:left+crop_w]
        
        if "target" in sample.keys():
            target = sample["target"]
            assert len(target.shape) == 3, "Expected a 3D array for the target"
            
            sample["target"] = target[:, top:top+crop_h, left:left+crop_w]
        
        return sample


class FilterClouds:
    def __init__(self, data_dir: str | Path, threshold = 0.1, with_datetime : bool = True):
        self.data_dir = data_dir
        self.threshold = threshold
        self.with_datetime = with_datetime

        csv_path = Path(data_dir) / "pastis_cloud_analysis.csv"
        
        if not os.path.isfile(csv_path):
            raise FileNotFoundError("'pastis_cloud_analysis.csv' not found in root data dir.")
        
        self.valid_indices = self._get_valid_indices(csv_path)
    
    def __call__(self, sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Keep only timestamps whose cloud percentage is below the threshold.
        """
        data, stem = sample["data"], sample["stem"]
        
        indices = self.valid_indices[stem]
        sample["data"] = data[indices]
        
        if self.with_datetime:
            dates = sample["dates"]
            sample["dates"] = dates[indices]
        
        return sample
    
    def _get_valid_indices(self, csv_path: Path) -> dict[str, list]:
        cloud_df = pd.read_csv(csv_path)
        valid_indices = {}
        
        for stem, group in cloud_df.groupby("stem"):
            indices_list = group[group["cloud_percentage"] < self.threshold]["timestamp"].tolist()
            valid_indices[stem] = indices_list
        
        return valid_indices