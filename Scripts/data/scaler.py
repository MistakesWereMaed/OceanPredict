import xarray as xr
import torch

class MinMaxScaler:
    def __init__(self, min_val=None, max_val=None):
        self.min_val = min_val
        self.max_val = max_val

    def fit_transform(self, ds: xr.Dataset):
        self.min_val = ds.min(dim=("time", "latitude", "longitude"))
        self.max_val = ds.max(dim=("time", "latitude", "longitude"))

        return (ds - self.min_val) / (self.max_val - self.min_val)

    def inverse_transform(self, arr: torch.Tensor, var_names: list[str]):
        min_vals = torch.tensor([self.min_val[v].values.item() for v in var_names], dtype=arr.dtype, device=arr.device)
        max_vals = torch.tensor([self.max_val[v].values.item() for v in var_names], dtype=arr.dtype, device=arr.device)

        shape = [1, -1] + [1] * (arr.ndim - 2)
        min_vals = min_vals.view(*shape)
        max_vals = max_vals.view(*shape)

        return arr * (max_vals - min_vals) + min_vals