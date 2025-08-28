import json
import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

import pytorch_lightning as pl

PATH_PARAMS = "../Models/FNO/params.json"

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat),
            requires_grad=False
        )

    def forward(self, x):
        B, C, H, W = x.shape

        with torch.amp.autocast(device_type="cuda", enabled=False):
            x_fp32 = x.float()
            x_ft = torch.fft.rfft2(x_fp32)

            out_ft = torch.zeros(
                B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device
            )

            out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1
            )

            x = torch.fft.irfft2(out_ft, s=(H, W))
            x = x.to(dtype=x_fp32.dtype)

        return x

class model(pl.LightningModule):
    def loss_fn(self, y_hat, y):
        return F.l1_loss(y_hat, y)
    
    def __init__(self, image_size, learning_rate, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.modes1 = int(kwargs["num_fourier_modes"])
        self.modes2 = int(kwargs["num_fourier_modes"])
        self.width = int(kwargs["fno_width"])
        self.depth = int(kwargs["num_fno_layers"])
        self.mlp_hidden_dim = int(kwargs["mlp_hidden_dim"])
        self.dropout = nn.Dropout(0.1)

        self.fc0 = nn.Conv2d(3, self.width, kernel_size=1)

        self.convs = nn.ModuleList([
            SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            for _ in range(self.depth)
        ])
        self.ws = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1)
            for _ in range(self.depth)
        ])

        self.fc1 = nn.Linear(self.width, self.mlp_hidden_dim)
        self.fc2 = nn.Linear(self.mlp_hidden_dim, 3)

        self.name = "FNO"

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        pad_h = (self.modes1 - H % self.modes1) % self.modes1
        pad_w = (self.modes2 - W % self.modes2) % self.modes2

        x = F.pad(x, [0, pad_w, 0, pad_h])  # Pad width (last), then height (first)
        x = self.fc0(x)

        for i, (conv, w) in enumerate(zip(self.convs, self.ws)):
            x1 = conv(x)
            x2 = w(x)
            x = torch.relu(x1 + x2)
            x = self.dropout(x)

        x = x.permute(0, 2, 3, 1)  # [B, H, W, width]
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)  # [B, out_channels, H, W]

        if pad_h > 0: x = x[..., :-pad_h, :]
        if pad_w > 0: x = x[..., :, :-pad_w]

        return x

    @staticmethod
    def get_sweep_congfig():
        return {
            "name": "FNO-Tuning",
            "method": "bayes",
            "metric": {
                "name": "val_loss/dataloader_idx_0",
                "goal": "minimize"
            },
            "parameters": {
                "batch_size": {"values": [1, 4, 8]},
                "num_fourier_modes": {"values": [8, 16, 24, 32]},
                "num_fno_layers": {"values": [1, 2, 4]},
                "fno_width": {"values": [32, 64, 96, 128]},
                "mlp_hidden_dim": {"values": [64, 128, 192, 256]},
                "learning_rate": {
                    "distribution": "log_uniform_values",
                    "min": 1e-6,
                    "max": 1e-3
                }
            }
        }
    
    @staticmethod
    def load_params():
        try:
            with open(PATH_PARAMS, "r") as f:
                params = json.load(f)
        except FileNotFoundError:
            print("No params found, using defaults")
            params = {
                "batch_size": 4,
                "num_fourier_modes": 20,
                "num_fno_layers": 2,
                "fno_width": 64,
                "mlp_hidden_dim": 128,
                "learning_rate": 1e-3,
            }
        return params
    

    
    def training_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        x = torch.cat((x[0], x[1]), dim=0)
        y = torch.cat((y[0], y[1]), dim=0)
        
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        
        y_hat = self.forward(x)
        val_loss = self.loss_fn(y_hat, y)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.forward(x)
        return {"preds": y_hat, "targets": y}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)