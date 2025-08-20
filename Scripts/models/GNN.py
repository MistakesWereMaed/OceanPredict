import json
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from torch_geometric.nn import GCNConv
from torch_geometric.utils import grid
from torch_geometric.data import Data, Batch

class model(pl.LightningModule):
    def __init__(self, image_size, learning_rate,**kwargs):
        super().__init__()
        self.save_hyperparameters()

        hidden_dim = int(kwargs["hidden_dim"])
        num_gnn_layers = int(kwargs["num_gnn_layers"])

        height, width = image_size
        self.H, self.W = height, width
        self.n_nodes = height * width
        self.dropout = 0.1

        # Define edge_index
        edge_index = grid(height, width)[0]

        self.input_proj = nn.Linear(3, hidden_dim)
        self.gnn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, 3)
        self.name = "GNN"
        self.edge_index = edge_index

    def forward(self, x):
        device = x.device
        edge_index = self.edge_index.to(device)

        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)  # [B, N, C]

        # Create a list of PyG Data objects, one per sample
        data_list = [
            Data(x=xb, edge_index=edge_index) for xb in x
        ]

        # Batch the graphs into one large graph
        batch = Batch.from_data_list(data_list).to(device)  # batch.x: [B*N, C]

        # Apply GNN layers on the full batch
        h = self.input_proj(batch.x)
        for conv in self.gnn_layers:
            h = conv(h, batch.edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        out = self.output_proj(h)  # [B*N, 3]

        # Unbatch and reshape
        out = out.view(B, self.n_nodes, -1)  # [B, N, 3]
        out = out.permute(0, 2, 1).reshape(B, -1, H, W)  # [B, 3, H, W]

        return out
    
    @staticmethod
    def get_hyperparam_space():
        return {
            "method": "bayes",
            "metric": {
                "name": "val_loss",
                "goal": "minimize"
            },
            "parameters": {
                "batch_size": {"values": [1, 4, 8, 16]},
                "hidden_dim": {"values": [128, 256, 384, 512]},
                "num_gnn_layers": {"values": [2, 3, 4]},
                "learning_rate": {
                    "min": 0.000001,
                    "max": 0.001,
                    "distribution": "log_uniform"
                }
            }
        }
    
    @staticmethod
    def load_params():
        try:
            with open("../../Params/GNN.json", "r") as f:
                params = json.load(f)
        except FileNotFoundError:
            params = {
                "batch_size": 8,
                "hidden_dim": 64,
                "num_gnn_layers": 1,
                "learning_rate": 1e-4,
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