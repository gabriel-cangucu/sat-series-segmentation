import lightning as L
import geopandas as gpd
import numpy as np
import json
from pathlib import Path
from typing import Callable, Any
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import Compose

from .transforms import (
    ToTensor,
    SampleRandomTimestamps,
    RandomFlip,
    RandomRotate,
    FilterClouds,
    Crop
)


class PASTIS_S2(Dataset):
    def __init__(
            self,
            data_dir: str | Path,
            normalize: bool = True,
            with_datetime: bool = True,
            transform: Callable[[dict], Any] | None = None
        ) -> None:
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.with_datetime = with_datetime
        self.transform = transform
        
        self.metadata = self._load_metadata(data_dir)
        self.mean_std = self._load_mean_std(data_dir)
    
    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        patch_id = self.metadata.iloc[idx]["ID_PATCH"]
        
        data = self._load_patch_data(patch_id).astype(np.float32)
        target = self._load_segmentation_annotation(patch_id).astype(np.int64)
        
        if self.normalize:
            data = self._normalize_data(data)
        
        sample = {
            "data": data,
            "target": target,
            "stem": "S2_" + str(patch_id)
        }
        
        if self.with_datetime:
            sample["dates"] = self._get_dates(idx).astype(np.float32)

        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _load_metadata(self, data_dir: str | Path) -> gpd.GeoDataFrame:
        """
        Load metadata from the dataset directory.
        """
        metadata_file_path = self.data_dir / "metadata.geojson"
        
        if not metadata_file_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_file_path}")
        
        metadata_df = gpd.read_file(metadata_file_path)
        metadata_df = metadata_df.reset_index(drop=True)
        
        return metadata_df
    
    def _load_mean_std(self, data_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """
        Load mean and standard deviation values for normalization.
        """
        mean_std_file_path = self.data_dir / "NORM_S2_patch.json"
        folds = list(range(1, 6))
        
        if not mean_std_file_path.exists():
            raise FileNotFoundError(f"Mean and standard deviation file not found at {mean_std_file_path}")
        
        with open(mean_std_file_path, "r") as f:
            norm_dict = json.load(f)
            
        mean = [norm_dict[f"Fold_{fold}"]["mean"] for fold in folds]
        mean = np.stack(mean).mean(axis=0).astype(np.float32)

        std = [norm_dict[f"Fold_{fold}"]["std"] for fold in folds]
        std = np.stack(std).mean(axis=0).astype(np.float32)
        
        return mean, std

    def _load_patch_data(self, patch_id: int) -> np.ndarray:
        patch_file_path = self.data_dir / "DATA_S2" / f"S2_{patch_id}.npy"
        patch_data = np.load(patch_file_path)
        
        return patch_data

    def _load_segmentation_annotation(self, patch_id: int) -> np.ndarray:
        target_file_path = self.data_dir / "ANNOTATIONS" / f"TARGET_{patch_id}.npy"
        target = np.load(target_file_path)
        
        return target
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        mean, std = self.mean_std
        
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        
        return (data - mean) / std
    
    def _get_dates(self, idx: int) -> np.ndarray:
        dates_dict = self.metadata.iloc[idx]["dates-S2"]
        dates_dict = json.loads(dates_dict)
        dates_dict = {k: dates_dict[k] for k in sorted(dates_dict.keys(), key=lambda x: int(x))}
        
        dates = [f"{str(date)[:4]}-{str(date)[4:6]}-{str(date)[6:]}" for date in dates_dict.values()]
        dates = [datetime.strptime(date, "%Y-%m-%d") for date in dates]
        dates = np.array([[date.timetuple().tm_year, date.timetuple().tm_yday] for date in dates])
        
        return dates
        

class PASTIS_S2_Module(L.LightningDataModule):
    """
    PASTIS dataset for Sentinel-2 data.
    """
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        
        self.batch_size = config.dataset.batch_size
        self.num_workers = config.dataset.num_workers
        self.num_timestamps = config.model.num_timestamps
        self.img_size = config.model.img_size
        self.train_data_dir = Path(config.dataset.train.data_dir)
        self.test_data_dir = Path(config.dataset.test.data_dir)

        self.transform_train = Compose([
            FilterClouds(self.train_data_dir, threshold=0.1),
            SampleRandomTimestamps(num_timestamps=self.num_timestamps),
            Crop(size=(self.img_size, self.img_size), crop_type="random"),
            RandomFlip(prob=0.5, orientation="hor"),
            RandomFlip(prob=0.5, orientation="ver"),
            RandomRotate(),
            ToTensor()
        ])
        self.transform_test = Compose([
            FilterClouds(self.train_data_dir, threshold=0.1),
            SampleRandomTimestamps(num_timestamps=self.num_timestamps),
            Crop(size=(self.img_size, self.img_size), crop_type="center"),
            ToTensor()
        ])

    def setup(self, stage: str | None = None) -> None:
        """
        Setup the dataset for training, validation, and testing.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = PASTIS_S2(self.train_data_dir, transform=self.transform_train)
        if stage == "test" or stage is None:
            self.test_dataset = PASTIS_S2(self.test_data_dir, transform=self.transform_test)

    def train_dataloader(self) -> Callable[[dict], DataLoader]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True
        )

    def test_dataloader(self) -> Callable[[dict], DataLoader]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        )