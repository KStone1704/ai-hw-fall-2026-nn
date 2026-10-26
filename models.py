"""Small MNIST models. Every forward method returns ten unnormalized logits."""

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(),
                                 nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        return self.net(x)


class PatchTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(1, 64, kernel_size=7, stride=7)
        self.cls = nn.Parameter(torch.zeros(1, 1, 64))
        self.position = nn.Parameter(torch.zeros(1, 17, 64))
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, dropout=0.1,
            activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2,
                                             enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 10))
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.position, std=0.02)
        # Encoder clones otherwise start with identical parameter values.
        for block in self.encoder.layers:
            for parameter in block.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)

    def forward(self, x):
        tokens = self.patch(x).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.cls.expand(x.shape[0], -1, -1), tokens), dim=1)
        return self.head(self.encoder(tokens + self.position)[:, 0])


def build_model(name):
    return {"mlp": MLP, "cnn": CNN, "transformer": PatchTransformer}[name]()
