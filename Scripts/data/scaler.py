import xarray as xr
import torch

class Scaler:
    def __init__(self, min_val=None, max_val=None):
        self.min_val = min_val
        self.max_val = max_val

    def fit_transform(self, ds: xr.Dataset):
        self.min_val = ds.min(dim=("time", "latitude", "longitude"))
        self.max_val = ds.max(dim=("time", "latitude", "longitude"))

        return 2 * (ds - self.min_val) / (self.max_val - self.min_val) - 1

    def inverse_transform(self, arr: torch.Tensor, var_names: list[str]):
        unscaled = []
        for i, var in enumerate(var_names):
            vmin = torch.tensor(self.min_val[var].values, dtype=arr.dtype, device=arr.device)
            vmax = torch.tensor(self.max_val[var].values, dtype=arr.dtype, device=arr.device)

            print(f"[Scaler] Variable: {var}")
            print(f"  Stored min: {vmin.item():.4f}, max: {vmax.item():.4f}")
            print(f"  Input scaled range: min={arr[:, i].min().item():.4f}, max={arr[:, i].max().item():.4f}")

            unscaled_var = (arr[:, i] + 1) / 2 * (vmax - vmin) + vmin   # [B, T, H, W]

            print(f"  Output unscaled range: min={unscaled_var.min().item():.4f}, max={unscaled_var.max().item():.4f}")

            unscaled.append(unscaled_var.unsqueeze(1))  # keep channel dim

        return torch.cat(unscaled, dim=1)  # [B, C, T, H, W]