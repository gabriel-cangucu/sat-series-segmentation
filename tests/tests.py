import torch
import lightning as L
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from data import PASTIS_S2, PASTIS_S2_Module, IBGE
from models import PrithviMAE
from data.transforms import FilterClouds


def test_pastis_data_loading() -> None:
    data_dir = "/storage/datasets/gabriel.lima/PASTIS"
    
    loader = PASTIS_S2(data_dir)
    sample = loader[0]
    
    assert len(loader) == 2433, "Missing data samples"
    assert sample["data"].shape[1:] == (10, 128, 128), "Data shape mismatch"
    assert sample["target"].shape == (3, 128, 128), "Target shape mismatch"


def test_pastis_data_module_transforms() -> None:
    data_dir = "/storage/datasets/gabriel.lima/PASTIS"
    
    config = OmegaConf.create({
        "model": {
            "name": "prithvi_mae",
            "img_size": 96,
            "num_channels": 10,
            "num_timestamps": 3
        },
        "dataset": {
            "batch_size": 16,
            "num_workers": 4,
            "train": {
                "name": "pastis",
                "data_dir": data_dir,
            },
            "test": {
                "name": "pastis",
                "data_dir": data_dir,
            }
        }
    })
    
    module = PASTIS_S2_Module(config)
    module.setup()
    loader = module.train_dataloader()
    
    sample = next(iter(loader))
    
    assert isinstance(sample["data"], torch.Tensor), "Data should be a tensor"
    assert isinstance(sample["target"], torch.Tensor), "Target should be a tensor"
    
    assert sample["data"].shape[2] == 3, "Data should have 3 timestamps"
    assert sample["dates"].shape[1] == 3, "Dates should have 3 timestamps"


def test_pastis_data_cloud_filtering() -> None:
    data_dir = "/storage/datasets/gabriel.lima/PASTIS"
    
    loader = PASTIS_S2(data_dir)
    sample = loader[0]
    
    num_timestamps_before = sample["data"].shape[0]
    sample = FilterClouds(data_dir, threshold=0.1)(sample)
    num_timestamps_after = sample["data"].shape[0]
    
    assert num_timestamps_after < num_timestamps_before, "Cloud filtering did not reduce timestamps"
    

def test_prithvi_model_output() -> None:
    config = OmegaConf.create({
        "model": {
            "name": "prithvi_mae",
            "img_size": 96,
            "num_channels": 10,
            "num_timestamps": 2
        }
    })
    
    batch_size = 16
    device = torch.device("cuda")

    model = PrithviMAE(config).to(device)

    x = torch.randn(batch_size, 10, 2, 96, 96).to(device)
    outputs = model(x)
    
    assert outputs["preds"].shape == (batch_size, 10, 2, 96, 96), "Output shape mismatch"


def test_prithvi_model_with_dates() -> None:
    def generate_random_dates(batch_size: int, num_timestamps: int = 3):
        days_of_year = torch.randint(1, 365 + 1, (batch_size, num_timestamps, 1))

        year_column = torch.full((batch_size, num_timestamps, 1), 2018)
        timestamp_tensor = torch.cat((year_column, days_of_year), dim=-1).float()

        return timestamp_tensor
    
    config = OmegaConf.create({
        "model": {
            "name": "prithvi_mae",
            "img_size": 128,
            "num_channels": 10,
            "num_timestamps": 3
        }
    })
    
    batch_size = 16
    device = torch.device("cuda")

    model = PrithviMAE(config).to(device)

    x = torch.randn(batch_size, 10, 3, 128, 128).to(device)
    dates = generate_random_dates(batch_size, num_timestamps=3).to(device)

    outputs = model(x, temporal_coords=dates)
    
    assert outputs["preds"].shape == (batch_size, 10, 3, 128, 128), "Output shape mismatch with dates"


def test_image_reconstruction() -> None:
    config = OmegaConf.create({
        "model": {
            "name": "prithvi_mae",
            "img_size": 128,
            "num_channels": 10,
            "num_timestamps": 3
        }
    })
    
    device = torch.device("cuda")
    model = PrithviMAE(config).to(device)
    
    x = torch.randn(16, 10, 3, 128, 128).to(device)
    patches = model.net.patchify(x)
    reconstructed = model.net.unpatchify(patches)
    
    assert reconstructed.shape == x.shape, "Reconstructed image shape mismatch"
    assert (x == reconstructed).all(), "Reconstructed image does not match original"


def test_ibge_data_loading() -> None:
    data_dir = "/storage/datasets/gabriel.lima/IBGE"
    
    loader = IBGE(data_dir)
    sample = loader[0]
    
    assert len(loader) == 50206, "Missing data samples"
    assert sample["data"].shape[1:] == (10, 128, 128), "Data shape mismatch"
    assert sample["stem"] is not None, "Stem should not be None"


if __name__ == "__main__":
    # test_pastis_data_loading()
    # test_pastis_data_module_transforms()
    # test_prithvi_model_output()
    # test_prithvi_model_with_dates()
    # test_pastis_data_cloud_filtering()
    # test_image_reconstruction()
    test_ibge_data_loading()
    
    print("All tests passed!")