import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

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

class model(nn.Module):
    def __init__(self, image_size, **kwargs):
        super().__init__()

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
    def get_hyperparam_space():
        return {
            "batch_size": hp.choice("batch_size", [1, 4, 8, 16]),
            "num_fourier_modes": hp.quniform("num_fourier_modes", 8, 32, 4),
            "num_fno_layers": hp.choice("num_fno_layers", [1, 2, 4]),
            "fno_width": hp.quniform("fno_width", 32, 128, 32),
            "mlp_hidden_dim": hp.quniform("mlp_hidden_dim", 64, 256, 64),
            "learning_rate": hp.loguniform("learning_rate", -6, -3),
        }