import xarray as xr
import torch

from torch.utils.data import Dataset, DataLoader
from . import scaler

class OceanDataset(Dataset):
    def __init__(self, path, scaler):
        raw_ds = xr.open_dataset(path)
        self.ds = scaler.fit_transform(raw_ds)

        input_vars = ['zos', 'u10', 'v10']
        target_vars = ['uo', 'vo', 'zos']

        x_list, y_list = [], []
        time_steps = self.ds.sizes["time"] - 1
        for i in range(time_steps):
            x = self.ds[input_vars].isel(time=i).to_array().values
            y = self.ds[target_vars].isel(time=i + 1).to_array().values
            
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
    _scaler = scaler.MinMaxScaler()
    dataset = OceanDataset(path, _scaler)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True
    )

    return dataloader, _scaler if return_scaler else dataloader
    
def get_image_size(path):
    ds = xr.open_dataset(path)

    lat_size = ds.sizes.get("latitude", 0)
    lon_size = ds.sizes.get("longitude", 0)

    return (lat_size, lon_size)
