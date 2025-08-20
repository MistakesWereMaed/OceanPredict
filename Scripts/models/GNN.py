import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.utils import grid
from torch_geometric.data import Data, Batch

class model(nn.Module):
    def __init__(self, image_size, **kwargs):
        super().__init__()
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
            "batch_size": hp.choice("batch_size", [1, 4, 8, 16]),
            "hidden_dim": hp.quniform("hidden_dim", 32, 256, 32),
            "num_gnn_layers": hp.choice("num_gnn_layers", [2, 3, 4]),
            "learning_rate": hp.loguniform("learning_rate", -6, -3),
        }