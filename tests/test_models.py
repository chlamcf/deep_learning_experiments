"""
Basic shape/sanity tests for the models. Run with: pytest tests/
"""
import torch

from src.models import Discriminator, Generator, MNISTCNN


def test_generator_output_shape():
    latent_dim = 100
    batch_size = 8
    gen = Generator(latent_dim=latent_dim)
    z = torch.randn(batch_size, latent_dim)
    out = gen(z)
    assert out.shape == (batch_size, 1, 28, 28)


def test_discriminator_output_shape():
    batch_size = 8
    disc = Discriminator()
    imgs = torch.randn(batch_size, 1, 28, 28)
    out = disc(imgs)
    assert out.shape == (batch_size,)
    assert torch.all((out >= 0) & (out <= 1))


def test_cnn_output_shape():
    batch_size = 8
    model = MNISTCNN(num_classes=10)
    imgs = torch.randn(batch_size, 1, 28, 28)
    out = model(imgs)
    assert out.shape == (batch_size, 10)
