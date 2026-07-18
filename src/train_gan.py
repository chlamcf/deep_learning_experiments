"""
Train a vanilla GAN on MNIST.

Usage:
    python -m src.train_gan --epochs 50 --batch-size 128 --lr 0.0002 --latent-dim 100
"""
import argparse
import csv
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from src.data import get_gan_dataloader
from src.models import Discriminator, Generator
from src.utils import get_logger, save_checkpoint, set_seed

logger = get_logger("train_gan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a vanilla GAN on MNIST")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs/gan_images")
    parser.add_argument("--checkpoint-dir", type=str, default="./outputs/gan_models")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                         help="Path prefix to resume from, e.g. outputs/gan_models (loads latest epoch checkpoints)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def find_latest_epoch(checkpoint_dir: str) -> int:
    """Scan checkpoint_dir for generator_epoch_N.pth and return the highest N, or -1 if none found."""
    if not os.path.isdir(checkpoint_dir):
        return -1
    epochs = []
    for fname in os.listdir(checkpoint_dir):
        if fname.startswith("generator_epoch_") and fname.endswith(".pth"):
            try:
                epochs.append(int(fname[len("generator_epoch_"):-len(".pth")]))
            except ValueError:
                continue
    return max(epochs) if epochs else -1


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_loader = get_gan_dataloader(args.data_root, args.batch_size)

    generator = Generator(latent_dim=args.latent_dim).to(device)
    discriminator = Discriminator().to(device)

    adversarial_loss = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    start_epoch = 0
    if args.resume:
        latest = find_latest_epoch(args.resume)
        if latest >= 0:
            generator.load_state_dict(torch.load(f"{args.resume}/generator_epoch_{latest}.pth", map_location=device))
            discriminator.load_state_dict(torch.load(f"{args.resume}/discriminator_epoch_{latest}.pth", map_location=device))
            start_epoch = latest + 1
            logger.info("Resumed from epoch %d", latest)
        else:
            logger.info("No checkpoints found in %s, starting from scratch", args.resume)

    loss_csv_path = os.path.join(args.output_dir, "loss_history.csv")
    write_header = not os.path.exists(loss_csv_path)
    csv_file = open(loss_csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow(["epoch", "batch", "d_loss", "g_loss"])

    for epoch in range(start_epoch, args.epochs):
        for i, (imgs, _) in enumerate(train_loader):
            real_imgs = imgs.to(device)
            real_labels = torch.ones(imgs.size(0), device=device)
            fake_labels = torch.zeros(imgs.size(0), device=device)

            optimizer_d.zero_grad()
            real_preds = discriminator(real_imgs)
            real_loss = adversarial_loss(real_preds, real_labels)

            z = torch.randn(imgs.size(0), args.latent_dim, device=device)
            fake_imgs = generator(z)
            fake_preds = discriminator(fake_imgs.detach())
            fake_loss = adversarial_loss(fake_preds, fake_labels)

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            z = torch.randn(imgs.size(0), args.latent_dim, device=device)
            gen_imgs = generator(z)
            gen_preds = discriminator(gen_imgs)
            g_loss = adversarial_loss(gen_preds, real_labels)
            g_loss.backward()
            optimizer_g.step()

            csv_writer.writerow([epoch, i, d_loss.item(), g_loss.item()])

            if i % args.log_interval == 0:
                logger.info(
                    "Epoch %d/%d Batch %d/%d D_loss %.4f G_loss %.4f",
                    epoch + 1, args.epochs, i, len(train_loader), d_loss.item(), g_loss.item(),
                )

        csv_file.flush()
        save_image(gen_imgs.data[:25], f"{args.output_dir}/epoch_{epoch}.png", nrow=5, normalize=True)
        save_checkpoint(generator.state_dict(), args.checkpoint_dir, f"generator_epoch_{epoch}.pth")
        save_checkpoint(discriminator.state_dict(), args.checkpoint_dir, f"discriminator_epoch_{epoch}.pth")

    csv_file.close()
    logger.info("Training complete. Loss history saved to %s", loss_csv_path)
    logger.info("Run `python -m src.plot_losses` to generate the loss curve plot.")


if __name__ == "__main__":
    train(parse_args())