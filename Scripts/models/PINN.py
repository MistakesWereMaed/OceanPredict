import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from linformer import LinformerSelfAttention

def calculate_seq_length(image_size, kernel_size, padding=0, stride=1):
    height_in, width_in = image_size
    kernel_height, kernel_width = kernel_size
    # Compute output height and width after Conv2D
    height_out = (height_in + 2 * padding - kernel_height) // stride + 1
    width_out = (width_in + 2 * padding - kernel_width) // stride + 1
    # Compute sequence length
    return height_out * width_out

class model(pl.LightningModule):
    def __init__(self, image_size, kernel_size, num_heads, embed_dim, linformer_k, mlp_hidden_dim, learning_rate, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        embed_dim = int(embed_dim)
        linformer_k = int(linformer_k)
        mlp_hidden_dim = int(mlp_hidden_dim)
        len = calculate_seq_length(image_size, kernel_size)
        # Data-Driven Components
        self.conv = nn.Conv2d(3, embed_dim, kernel_size=kernel_size)
        self.attn = LinformerSelfAttention(dim=embed_dim, seq_len=len, k=linformer_k, heads=num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU()
        )
        self.transconv = nn.ConvTranspose2d(mlp_hidden_dim, 3, kernel_size=kernel_size)
        # Physics-Informed Components
        self.g = 9.81
        self.f = 1e-4
        self.name = "PINN"

    def forward(self, x):
        # Step 1: Data-Driven Computation
        x = self.conv(x)
        batch_size, channels, height, width = x.shape
        seq_len = height * width
        x = x.view(batch_size, channels, seq_len).transpose(1, 2)
        x = self.attn(x, context=x)
        x = self.norm(x)
        x = self.mlp(x)
        x = x.permute(0, 2, 1).view(batch_size, -1, height, width)
        x = self.transconv(x)
        x = torch.clamp(x, min=1e-5, max=1e5)
        # Extract u', v', SSH from Data-Driven output
        u_prime, v_prime, ssh = torch.chunk(x, chunks=3, dim=1)
        u_prime = u_prime.squeeze(1)
        v_prime = v_prime.squeeze(1)
        ssh = ssh.squeeze(1)
        # Step 2: Physics-Informed Computation (∇SSH → geostrophic velocity)
        dudx = torch.diff(ssh, dim=-1, append=ssh[:, :, -1:])
        dvdy = torch.diff(ssh, dim=-2, append=ssh[:, -1:, :])
        u_g = self.g / self.f * dvdy
        v_g = -self.g / self.f * dudx
        # Step 3: Sum Module (Combining Data-Driven and Physics-Informed velocities)
        u = u_g + u_prime
        v = v_g + v_prime

        return torch.stack((u, v, ssh), dim=1)

    @staticmethod
    def get_sweep_congfig():
        return {
            "name": "PINN-Tuning",
            "method": "bayes",
            "metric": {
                "name": "val_loss/dataloader_idx_0",
                "goal": "minimize"
            },
            "parameters": {
                "batch_size": {"values": [1, 4, 8]},
                "kernel_size": {"values": [[3, 3], [5, 10], [7, 7]]},
                "num_heads": {"values": [1, 2, 4]},
                "embed_dim": {"values": [128, 256, 384, 512]},
                "linformer_k": {"values": [32, 64, 96, 128]},
                "mlp_hidden_dim": {"values": [32, 64, 96, 128]},
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
            with open("../Params/PINN.json", "r") as f:
                params = json.load(f)
        except FileNotFoundError:
            print("No params found, using defaults")
            params = {
                "batch_size": 8,
                "kernel_size": (3, 3),
                "num_heads": 1,
                "embed_dim": 128,
                "linformer_k": 128,
                "mlp_hidden_dim": 128,
                "learning_rate": 1e-5,
            }
        return params
    

    
    def training_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        x = torch.cat((x[0], x[1]), dim=0)
        y = torch.cat((y[0], y[1]), dim=0)
        
        y_hat = self.forward(x)
        loss = F.l1_loss(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        
        y_hat = self.forward(x)
        val_loss = F.l1_loss(y_hat, y)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.forward(x)
        return {"preds": y_hat, "targets": y}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)