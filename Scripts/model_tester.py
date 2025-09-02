import argparse
import xarray as xr
import numpy as np

import pytorch_lightning as pl

from models._selector import select_model
from models._tester_module import OceanTestModule
from data.scaler import Scaler
from data.data_loader import load_test_data

PROJECT = "OceanPredict"
PATH_LOGS = "../Logs"

PATH_TEST = "../Data/test.nc"
PATH_MODELS = "../Models"

TARGET_DAYS = 15
TARGET_VARS = ["uo", "vo", "zos"]

LON_MIN = 120
LON_MAX = 170
LAT_MIN = 28
LAT_MAX = 43

def save_results(model_type, loss, predictions, targets):
    results_path = f"{PATH_MODELS}/{model_type}/results.nc"

    preds_np = predictions.squeeze(0).numpy().astype(np.float32)  # [C, T, H, W]
    targets_np = targets.squeeze(0).numpy().astype(np.float32)    # [C, T, H, W]
    C, T, H, W = preds_np.shape

    ds = xr.Dataset(
        {
            "loss": (("lead_time",), loss.astype(np.float32)),
            "predictions": (("channel", "time", "latitude", "longitude"), preds_np),
            "targets": (("channel", "time", "latitude", "longitude"), targets_np),
        },
        coords={
            "channel": TARGET_VARS,
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

    scaler = Scaler()
    dataloader = load_test_data(PATH_TEST, scaler, TARGET_DAYS)

    model = select_model(model_type)
    model = model.load_from_checkpoint(f"{PATH_MODELS}/{model_type}/model.ckpt")

    test_module = OceanTestModule(model, model.loss_fn, scaler, TARGET_DAYS, TARGET_VARS)

    trainer = pl.Trainer(accelerator="gpu", devices=1, precision="16-mixed", logger=False)
    trainer.predict(test_module, dataloader)

    predictions = test_module.results["predictions"]
    targets = test_module.results["targets"]
    loss = test_module.results["losses"]

    save_results(model_type, loss, predictions, targets)

if __name__ == "__main__":
    main()