import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import imageio
import shutil


class Normal:

    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.sqrt(std)
        self.dim = len(mean)

    def sample(self):
        z = np.array([random.gauss(0, 1) for _ in range(self.dim)])
        return self.mean + z * self.std


class MoG():

    def __init__(self, weights, params):
        self.weights = weights
        self.params = params
        self.fns = [Normal(mean=mean, std=std) for (mean, std) in self.params]

    def sample(self, n_samples=1):
        samples = []
        for _ in range(n_samples):
            fn = random.choices(self.fns, self.weights)[0]
            samples.append(fn.sample())
        return samples


class Generator(nn.Module):

    def __init__(self, latent_dim=2, n_hidden_layers=2, hidden_dim=256, out_dim=2):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.BatchNorm1d(hidden_dim))

        layers.append(nn.Linear(hidden_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):

    def __init__(self, in_dim=2, n_hidden_layers=2, hidden_dim=256, out_dim=1):
        super(Discriminator, self).__init__()

        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(0.3))

        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.3))

        layers.append(nn.Linear(hidden_dim, out_dim))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def create_video_from_frames(frames_dir, output_path, fps=10):
    frames_dir = Path(frames_dir)
    frame_files = sorted(frames_dir.glob("frame_*.png"), key=lambda x: int(x.stem.split('_')[1]))

    if not frame_files:
        print("No frames found to create video")
        return

    print(f"Creating video from {len(frame_files)} frames...")

    images = []
    for frame_file in frame_files:
        images.append(imageio.imread(frame_file))

    imageio.mimsave(output_path, images, format='FFMPEG', fps=fps, codec='libx264')
    print(f"Video saved to: {output_path}")


class GAN:

    def __init__(self, gen=None, disc=None):
        self.gen = gen
        self.disc = disc
        self.criterion = nn.BCELoss()

    def step_gen(self, x_fake, label_real):
        pred_fake = self.disc(x_fake)
        loss = self.criterion(pred_fake, label_real)
        return loss

    def step_disc(self, x_real, x_fake, label_real, label_fake):
        pred_real = self.disc(x_real)
        loss_real = self.criterion(pred_real, label_real)

        pred_fake = self.disc(x_fake.detach())
        loss_fake = self.criterion(pred_fake, label_fake)

        loss = loss_real + loss_fake
        return loss


class MoGData:

    def __init__(self, mog=None, size=1024):
        self.mog = mog
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sample = self.mog.sample(n_samples=1)[0]
        return torch.tensor(sample, dtype=torch.float32)


def run_training(epocs=100, iterations=500, batch_size=128, latent_dim=2, hidden_dim=256, n_hidden_layers=3, lr=0.0002,
                 create_video=False, frame_interval=1, video_fps=10, n_video_samples=500):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    mog = MoG(weights=[0.75, 0.25], params=[((10, 10), (3, 1.4)), ((0, 0), (1, 2))])

    dataset = DataLoader(MoGData(mog=mog, size=iterations * batch_size), batch_size=batch_size, shuffle=True)

    generator = Generator(latent_dim=latent_dim, n_hidden_layers=n_hidden_layers, hidden_dim=hidden_dim, out_dim=2).to(
        device)
    discriminator = Discriminator(in_dim=2, n_hidden_layers=n_hidden_layers, hidden_dim=hidden_dim, out_dim=1).to(
        device)

    gan = GAN(gen=generator, disc=discriminator)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    history = {'d_loss': [], 'g_loss': [], 'epoc': []}

    Path("GAN/outputs").mkdir(exist_ok=True)

    frames_dir = Path("GAN/outputs/frames")
    if create_video:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(exist_ok=True)
        print(f"Video generation enabled: saving frames every {frame_interval} batches")
        frame_counter = 0

    print(f"Starting training: {epocs} epochs, {iterations} iterations per epoch, batch size {batch_size}")

    for epoc in tqdm(range(epocs), ascii=True, unit='epoc'):
        d_losses = []
        g_losses = []

        for batch_idx, x_real in enumerate(tqdm(dataset, ascii=True, unit='it', leave=False)):
            x_real = x_real.to(device)
            batch_size_actual = x_real.size(0)

            label_real = torch.ones(batch_size_actual, 1).to(device)
            label_fake = torch.zeros(batch_size_actual, 1).to(device)

            optimizer_D.zero_grad()

            z = torch.randn(batch_size_actual, latent_dim).to(device)
            x_fake = generator(z)

            loss_D = gan.step_disc(x_real, x_fake, label_real, label_fake)
            loss_D.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            z = torch.randn(batch_size_actual, latent_dim).to(device)
            x_fake = generator(z)

            loss_G = gan.step_gen(x_fake, label_real)
            loss_G.backward()
            optimizer_G.step()

            d_losses.append(loss_D.item())
            g_losses.append(loss_G.item())

            if create_video and (batch_idx % frame_interval == 0):
                generator.eval()
                with torch.no_grad():
                    z_vis = torch.randn(n_video_samples, latent_dim).to(device)
                    fake_samples_vis = generator(z_vis).cpu().numpy()
                    real_samples_vis = np.array([mog.sample(1)[0] for _ in range(n_video_samples)])

                    plt.figure(figsize=(8, 8))
                    plt.scatter(real_samples_vis[:, 0], real_samples_vis[:, 1],
                                alpha=0.5, s=10, label='Real (MoG)', c='blue')
                    plt.scatter(fake_samples_vis[:, 0], fake_samples_vis[:, 1],
                                alpha=0.5, s=10, label='Synthetic (GAN)', c='red')
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.title(f'Epoch {epoc + 1}/{epocs} - Batch {batch_idx + 1}')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.xlim(-5, 15)
                    plt.ylim(-5, 15)

                    frame_path = frames_dir / f"frame_{frame_counter:05d}.png"
                    plt.savefig(frame_path, dpi=100, bbox_inches='tight')
                    plt.close()
                    frame_counter += 1

                generator.train()

        avg_d_loss = np.mean(d_losses)
        avg_g_loss = np.mean(g_losses)
        history['d_loss'].append(avg_d_loss)
        history['g_loss'].append(avg_g_loss)
        history['epoc'].append(epoc)

        if (epoc + 1) % 10 == 0:
            print(f"\nEpoch [{epoc + 1}/{epocs}] - D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")

    print("\nTraining completed!")

    print("Generating samples and creating plots...")

    generator.eval()
    with torch.no_grad():
        n_samples = 2000
        z = torch.randn(n_samples, latent_dim).to(device)
        fake_samples = generator(z).cpu().numpy()

        real_samples = np.array([mog.sample(1)[0] for _ in range(n_samples)])

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(real_samples[:, 0], real_samples[:, 1], alpha=0.5, s=10, label='Real (MoG)', c='blue')
    plt.scatter(fake_samples[:, 0], fake_samples[:, 1], alpha=0.5, s=10, label='Synthetic (GAN)', c='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Real vs Synthetic Samples')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history['epoc'], history['d_loss'], label='Discriminator Loss', alpha=0.7)
    plt.plot(history['epoc'], history['g_loss'], label='Generator Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('GAN/outputs/gan_results.png', dpi=150)
    print("Saved plot to GAN/outputs/gan_results.png")
    plt.close()

    torch.save(generator.state_dict(), 'GAN/outputs/generator.pth')
    torch.save(discriminator.state_dict(), 'GAN/outputs/discriminator.pth')
    print("Saved models to GAN/outputs/")

    if create_video:
        video_path = 'GAN/outputs/training_progress.mp4'
        create_video_from_frames(frames_dir, video_path, fps=video_fps)

    return history, generator, discriminator, mog


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train GAN on mixture of gaussians")
    parser.add_argument('-e', '--epocs', dest='epocs', type=int, default=100,
                        help='Number of epochs to run (default: 100)')
    parser.add_argument('-i', '--iterations', dest='iterations', type=int, default=500,
                        help='Number of iterations per epoch (default: 500)')
    parser.add_argument('-b', '--batch', dest='batch_size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--latent-dim', dest='latent_dim', type=int, default=2, help='Latent dimension (default: 2)')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=256,
                        help='Hidden layer dimension (default: 256)')
    parser.add_argument('--n-hidden', dest='n_hidden_layers', type=int, default=3,
                        help='Number of hidden layers (default: 3)')
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='Learning rate (default: 0.0002)')
    parser.add_argument('--create-video', dest='create_video', action='store_true',
                        help='Create training progress video')
    parser.add_argument('--frame-interval', dest='frame_interval', type=int, default=10,
                        help='Save frame every N batches (default: 10)')
    parser.add_argument('--video-fps', dest='video_fps', type=int, default=10,
                        help='Video frames per second (default: 10)')
    parser.add_argument('--n-video-samples', dest='n_video_samples', type=int, default=500,
                        help='Number of samples per video frame (default: 500)')

    args = parser.parse_args()

    run_training(
        epocs=args.epocs,
        iterations=args.iterations,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden_layers,
        lr=args.lr,
        create_video=args.create_video,
        frame_interval=args.frame_interval,
        video_fps=args.video_fps,
        n_video_samples=args.n_video_samples
    )
