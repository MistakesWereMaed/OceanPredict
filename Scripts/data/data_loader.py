import xarray as xr
import torch

from torch.utils.data import Dataset, DataLoader
from .scaler import Scaler

INPUT_VARS = ["zos", "u10", "v10"]
TARGET_VARS = ["uo", "vo", "zos"]

class OceanDataset(Dataset):
    def __init__(self, path, scaler):
        raw_ds = xr.open_dataset(path)
        self.ds = scaler.fit_transform(raw_ds)

        x_list, y_list = [], []
        time_steps = self.ds.sizes["time"] - 1
        for i in range(time_steps):
            x = self.ds[INPUT_VARS].isel(time=i).to_array().values
            y = self.ds[TARGET_VARS].isel(time=i + 1).to_array().values
            
            x_tensor = torch.tensor(x, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)

            x_list.append(x_tensor)
            y_list.append(y_tensor)

        self.x = torch.stack(x_list)
        self.y = torch.stack(y_list)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def load_data(path, batch_size, num_workers=2, return_scaler=False):
    _scaler = Scaler()
    dataset = OceanDataset(path, _scaler)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True
    )

    return dataloader, _scaler if return_scaler else dataloader
    
def get_image_size(path):
    ds = xr.open_dataset(path)

    lat_size = ds.sizes.get("latitude", 0)
    lon_size = ds.sizes.get("longitude", 0)

    return (lat_size, lon_size)

def load_test_data(path_test, scaler, target_days):
    raw_ds = xr.open_dataset(path_test)
    ds = scaler.fit_transform(raw_ds)
    # stack inputs [T, C, H, W]
    inputs = torch.stack([torch.tensor(ds[var].values).float() for var in INPUT_VARS], dim=1)
    targets = torch.stack([torch.tensor(ds[var].values).float() for var in TARGET_VARS], dim=1)

    input_t0 = inputs[0]                        # [C, H, W], only first day
    target_sequence = targets[1 : 1 + target_days]  # [T, C, H, W], next 15 days
    target_sequence = target_sequence.permute(1, 0, 2, 3)  # [C, T, H, W]
    # add batch dimension
    input_t0 = input_t0.unsqueeze(0)                # [B=1, C, H, W]
    target_sequence = target_sequence.unsqueeze(0)  # [B=1, C, T, H, W]

    dataset = torch.utils.data.TensorDataset(input_t0, target_sequence)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    return dataloader