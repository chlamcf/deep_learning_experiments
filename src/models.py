"""
Model definitions for the MNIST GAN and CNN classifier.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """Fully-connected generator for a vanilla GAN on MNIST (28x28 grayscale)."""

    def __init__(self, latent_dim: int = 100, img_channels: int = 1, img_size: int = 28):
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, img_channels * img_size * img_size),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        img = self.model(z)
        img = img.view(img.size(0), self.img_channels, self.img_size, self.img_size)
        return img


class Discriminator(nn.Module):
    """Fully-connected discriminator for a vanilla GAN on MNIST (28x28 grayscale)."""

    def __init__(self, img_channels: int = 1, img_size: int = 28):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_channels * img_size * img_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        flattened = img.view(img.size(0), -1)
        validity = self.model(flattened)
        return validity.squeeze(1)


class MNISTCNN(nn.Module):
    """Simple CNN classifier for MNIST digits."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
