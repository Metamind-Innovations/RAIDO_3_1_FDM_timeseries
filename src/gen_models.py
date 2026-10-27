"""
gen_models.py
Generator and Discriminator for 1D synthetic time series generation.

Generator    : latent z (batch, 128) → synthetic series (batch, 512, 1)
Discriminator: series (batch, 512, 1) → real/fake score (batch, 1)
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    1D CNN decoder.
    Input : latent vector z of shape (batch, latent_dim)
    Output: synthetic series of shape (batch, seq_out, 1)
            values scaled to [rh_min, rh_max] via Sigmoid
            (the aggregator sets the range from the dataset setup)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        seq_out: int = 512,
        rh_min: float = 0.0,
        rh_max: float = 100.0,
    ):
        super().__init__()
        self.seq_out = seq_out
        self.rh_min = rh_min
        self.rh_max = rh_max

        # Project latent → (batch, 256, 8) seed sequence
        self.fc = nn.Linear(latent_dim, 256 * 8)

        self.deconv = nn.Sequential(
            # (batch, 256, 8) → (batch, 128, 16)
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # (batch, 128, 16) → (batch, 64, 32)
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # (batch, 64, 32) → (batch, 32, 64)
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            # (batch, 32, 64) → (batch, 16, 128)
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            # (batch, 16, 128) → (batch, 1, 512)
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=4, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, latent_dim)
        Returns:
            series: (batch, seq_out, 1)  values in [rh_min, rh_max]
        """
        x = self.fc(z)                          # (batch, 256*8)
        x = x.view(z.size(0), 256, 8)          # (batch, 256, 8)
        x = self.deconv(x)                      # (batch, 1, 512)
        x = x.permute(0, 2, 1)                  # (batch, 512, 1)
        x = x * (self.rh_max - self.rh_min) + self.rh_min
        return x


class Discriminator(nn.Module):
    """
    1D CNN classifier — stays local on each client.
    Input : series (batch, 512, 1)
    Output: real/fake score (batch, 1)  in [0, 1]
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # (batch, 1, 512) → (batch, 16, 254)
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            # → (batch, 32, 125)
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            # → (batch, 64, 61)
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            # → (batch, 64, 1)
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 512, 1)
        Returns:
            score: (batch, 1)
        """
        # Conv1d expects (batch, channels, length)
        x = x.permute(0, 2, 1)   # (batch, 1, 512)
        x = self.net(x)           # (batch, 64, 1)
        return self.head(x)       # (batch, 1)
