import argparse
import wandb
import json
import os

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from models._selector import select_model, initialize_model
from data.data_loader import load_data, get_image_size

PATH_TRAIN = "../Data/train.nc"
PATH_VAL = "../Data/val.nc"
PATH_PARAMS = "../Params"
PATH_SECRETS = "data/secrets.json"
PATH_LOGS = "../Logs"

PROJECT = "OceanPredict"
MAX_EPOCHS = 1

def train(model_type, config=None):
    config = config or {}
    logger = WandbLogger(project=PROJECT, name=f'{model_type}-tuning', config=config, save_dir=PATH_LOGS)

    size = get_image_size(PATH_TRAIN)
    model, batch_size = initialize_model(model_type, size, config)

    train_loader = load_data(PATH_TRAIN, batch_size=batch_size)
    val_loader = load_data(PATH_VAL, batch_size=batch_size)

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        logger=logger,
        log_every_n_steps=10,
        enable_checkpointing=False
    )

    trainer.fit(model, train_loader, val_loader)

def save_best_config(entity, sweep_id, model_type):
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{PROJECT}/{sweep_id}")

    runs = sweep.runs
    best_run = min(runs, key=lambda run: run.summary.get("val_loss/dataloader_idx_0", float("inf")))

    best_config = dict(best_run.config)
    output_file = f"{PATH_PARAMS}/{model_type}.json"

    with open(output_file, "w") as f:
        json.dump(best_config, f, indent=4)

    print(best_config)

def main():
    os.environ["WANDB_DIR"] = PATH_LOGS
    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")

    parser.add_argument("--model", type=str, required=True, help="Type of model")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials")
    args = parser.parse_args()

    model_type = args.model
    trials = args.trials

    model_class = select_model(model_type)
    sweep_config = model_class.get_sweep_congfig()

    with open(PATH_SECRETS, 'r') as f:
        secrets = json.load(f)

    entity = secrets["WANDB_ENTITY"]
    
    sweep_id = wandb.sweep(sweep_config, project=PROJECT, entity=entity)
    wandb.agent(sweep_id, function=lambda: train(model_type=model_type), count=trials)

    #save_best_config(entity, sweep_id, model_type)

if __name__ == "__main__":
    main()