import torch
import numpy as np
import pytorch_lightning as pl

def rescale(x, dim=(0, 2, 3, 4)):
    x_min = x.amin(dim=dim, keepdim=True)
    x_max = x.amax(dim=dim, keepdim=True)
    
    return 2 * (x - x_min) / (x_max - x_min + 1e-8) - 1

class OceanTestModule(pl.LightningModule):
    def __init__(self, model, loss_fn, scaler, target_days, target_vars):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.scaler = scaler

        self.target_days = target_days
        self.target_vars = target_vars

        # storage for evaluation
        self.all_predictions = []
        self.all_targets = []

    def predict_step(self, batch, batch_idx):
        x_t0, target_sequence = batch  # [B=1, C, H, W], [B=1, C, T, H, W]

        preds = []
        input_t = x_t0
        for t in range(self.target_days):
            output = self.model(input_t)        # [B, C, H, W]
            preds.append(output.unsqueeze(2))   # [B, C, 1, H, W]
            input_t = output.detach()           # autoregression

        preds = torch.cat(preds, dim=2)  # [B, C, T, H, W]

        self.all_predictions.append(preds.cpu())
        self.all_targets.append(target_sequence.cpu())

        return preds

    def on_predict_epoch_end(self):
        predictions_scaled = torch.cat(self.all_predictions, dim=0)  # [1, C, T, H, W]
        targets_scaled = torch.cat(self.all_targets, dim=0)          # [1, C, T, H, W]
        predictions_scaled = rescale(predictions_scaled)
        targets_scaled = rescale(targets_scaled)
        # Unscale to original units
        predictions = self.scaler.inverse_transform(predictions_scaled.cpu(), self.target_vars)
        targets = self.scaler.inverse_transform(targets_scaled.cpu(), self.target_vars)
        # Compute daily losses
        B, C, T, H, W = predictions.shape
        daily_losses = []
        for t in range(T):
            loss_t = self.loss_fn(predictions[:, :, t], targets[:, :, t])
            daily_losses.append(loss_t.item())
        daily_losses = np.array(daily_losses)

        print(f"Average loss by lead time: {daily_losses}")

        self.results = {
            "predictions": predictions,
            "targets": targets,
            "losses": daily_losses,
        }