import argparse
import xarray as xr
import numpy as np

import torch
import pytorch_lightning as pl

from models._selector import select_model
from data.scaler import MinMaxScaler

PROJECT = "OceanPredict"
PATH_LOGS = "../Logs"

PATH_TEST = "../Data/test.nc"
PATH_MODELS = "../Models"

TARGET_DAYS = 15
INPUT_VARS = ["zos", "u10", "v10"]
TARGET_VARS = ["uo", "vo", "zos"]

LON_MIN = 10
LON_MAX = 38
LAT_MIN = -45
LAT_MAX = -32

class OceanTestModule(pl.LightningModule):
    def __init__(self, model, loss_fn, scaler):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.scaler = scaler

        # storage for evaluation
        self.all_predictions = []
        self.all_targets = []

    def predict_step(self, batch, batch_idx):
        x_t0, target_sequence = batch

        preds = []
        input_t = x_t0
        for t in range(TARGET_DAYS):
            output = self.model(input_t)        # [B, C, H, W]
            preds.append(output.unsqueeze(2))   # accumulate [B, C, 1, H, W]
            # next step autoregression (still scaled)
            input_t = output.detach()

        preds = torch.cat(preds, dim=2)  # [B, C, T, H, W]

        # store raw scaled tensors
        self.all_predictions.append(preds.cpu())
        self.all_targets.append(target_sequence.cpu())

        return preds

    def on_predict_epoch_end(self):
        # concat all batches (still scaled)
        predictions_scaled = torch.cat(self.all_predictions, dim=0)  # [B, C, T, H, W]
        targets_scaled = torch.cat(self.all_targets, dim=0)          # [B, C, T, H, W]

        # unscale once, on CPU
        predictions = self.scaler.inverse_transform(predictions_scaled.cpu(), TARGET_VARS)
        targets = self.scaler.inverse_transform(targets_scaled.cpu(), TARGET_VARS)

        # compute daily losses in unscaled space
        B, C, T, H, W = predictions.shape
        daily_losses = []
        for t in range(T):
            loss_t = self.loss_fn(predictions[:, :, t], targets[:, :, t])
            daily_losses.append(loss_t.item())
        daily_losses = np.array(daily_losses)

        print(f"Average loss by lead time: {daily_losses}")

        # stash for outside access
        self.results = {
            "predictions": predictions,
            "targets": targets,
            "losses": daily_losses,
        }

def make_test_dataset(path_test, scaler):
    raw_ds = xr.open_dataset(path_test)
    ds = scaler.fit_transform(raw_ds)
    # stack inputs [T, C, H, W]
    inputs = torch.stack([torch.tensor(ds[var].values).float() for var in INPUT_VARS], dim=1)
    targets = torch.stack([torch.tensor(ds[var].values).float() for var in TARGET_VARS], dim=1)

    total_days = inputs.shape[0]
    usable_days = total_days - TARGET_DAYS

    input_t0 = inputs[:usable_days]  # [B, C, H, W]
    target_sequence = torch.stack(
        [targets[t + 1 : t + 1 + TARGET_DAYS] for t in range(usable_days)]
    )  # [B, T, C, H, W]
    target_sequence = target_sequence.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]

    dataset = torch.utils.data.TensorDataset(input_t0, target_sequence)
    return dataset

def save_results(model_type, loss, predictions, targets):
    results_path = f"{PATH_MODELS}/{model_type}/results.nc"

    preds_np = predictions.numpy().astype(np.float32)
    targets_np = targets.numpy().astype(np.float32)
    B, C, T, H, W = preds_np.shape

    ds = xr.Dataset(
        {
            "loss": (("lead_time",), loss.astype(np.float32)),
            "predictions": (("sample", "channel", "time", "latitude", "longitude"), preds_np),
            "targets": (("sample", "channel", "time", "latitude", "longitude"), targets_np),
        },
        coords={
            "sample": np.arange(B),
            "channel": np.arange(C),
            "time": np.arange(T),
            "latitude": np.linspace(LAT_MIN, LAT_MAX, H),
            "longitude": np.linspace(LON_MIN, LON_MAX, W),
        },
    )

    ds.to_netcdf(results_path)

def main():
    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")

    parser.add_argument("--model", type=str, required=True, help="Type of model")
    args = parser.parse_args()
    model_type = args.model

    scaler = MinMaxScaler()
    dataset = make_test_dataset(PATH_TEST, scaler)

    model = select_model(model_type)
    model = model.load_from_checkpoint(f"{PATH_MODELS}/{model_type}/model.ckpt")

    test_module = OceanTestModule(model, model.loss_fn, scaler)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    trainer = pl.Trainer(accelerator="gpu", devices=1, precision="16-mixed", logger=False)
    trainer.predict(test_module, dataloader)

    predictions = test_module.results["predictions"]
    targets = test_module.results["targets"]
    loss = test_module.results["losses"]

    save_results(model_type, loss, predictions, targets)

if __name__ == "__main__":
    main()