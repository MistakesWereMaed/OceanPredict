import argparse

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from models._selector import initialize_model
from data.data_loader import load_data, get_image_size

PATH_TRAIN = "../Data/train.nc"
PATH_TEST = "../Data/test.nc"

PROJECT = "OceanPredict"
PATH_MODELS = "../Models"
PATH_LOGS = "../Logs"

def main():
    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")

    parser.add_argument("--model", type=str, required=True, help="Type of model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    args = parser.parse_args()

    epochs = args.epochs
    model_type = args.model

    size = get_image_size(PATH_TRAIN)
    model, batch_size = initialize_model(model_type, size)

    logger = WandbLogger(name=f"{model_type}-Training", project=PROJECT, save_dir=PATH_LOGS)
    checkpoint_cb = ModelCheckpoint(monitor="val_loss/dataloader_idx_0", save_top_k=1, mode="min", filename="model", dirpath=f"{PATH_MODELS}/{model.name}")
    early_stop_cb = EarlyStopping(monitor="val_loss/dataloader_idx_0", patience=3, mode="min")

    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        logger=logger,
        callbacks=[checkpoint_cb, early_stop_cb],
        log_every_n_steps=10
    )

    train_loader = load_data(PATH_TRAIN, batch_size=batch_size)
    val_loader = load_data(PATH_TEST, batch_size=batch_size)

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()