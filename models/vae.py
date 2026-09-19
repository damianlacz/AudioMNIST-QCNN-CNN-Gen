import torch as t
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import librosa

from torch.utils.data import DataLoader
from tqdm import tqdm

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        #nn.init.uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        #nn.init.uniform_(self.conv2.weight)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        #nn.init.uniform_(self.conv3.weight)
        self.flatten = nn.Flatten(1, -1)

        self.bneck1 = nn.Linear(7040, 512)
        self.bneck2 = nn.Linear(512, latent_dim)

        self.mu = nn.Linear(latent_dim, latent_dim)
        self.logvar = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.flatten(x)

        x = self.bneck1(x)
        x = self.bneck2(x)

        return self.mu(x), self.logvar(x)

class Decoder(nn.Module):
  def __init__(self, latent_dim, embed_dim=64):
      super().__init__()
      self.bneck = nn.Linear(latent_dim + embed_dim, 64 * 10 * 11)
      nn.init.normal_(self.bneck.weight)
      self.unflatten = nn.Unflatten(1, (64, 10, 11))

      self.trans_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
      self.trans_conv2 = nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      self.trans_conv3 = nn.ConvTranspose2d(16, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      self.out_conv = nn.ConvTranspose2d(1, 1, kernel_size=(2, 1), stride=(1, 1), padding=(1, 1))

      self.leaky_relu = nn.LeakyReLU()
      self.relu = nn.ReLU()

  def forward(self, x, emb):
    x = t.cat([x, emb], dim=1)
    x = self.bneck(x)
    x = self.unflatten(x)

    x = self.leaky_relu(self.trans_conv1(x))
    x = self.leaky_relu(self.trans_conv2(x))
    x = self.leaky_relu(self.trans_conv3(x))

    x = self.relu(self.out_conv(x))

    return x

class VAE(nn.Module):
  def __init__(self, latent_dim=128, num_classes=10, embed_dim=64):
      super().__init__()
      self.latent_dim = latent_dim
      self.num_classes = num_classes

      self.num_embed = nn.Embedding(num_embeddings=num_classes, embedding_dim=embed_dim)

      self.encoder = Encoder(latent_dim)
      self.decoder = Decoder(latent_dim, embed_dim)

  def reparameterize(self, mu, logvar):
      std = t.exp(0.5 * logvar)
      eps = t.randn_like(std)
      return mu + eps * std

  def forward(self, x, label):
      mu, logvar = self.encoder(x)
      z = self.reparameterize(mu, logvar)

      emb = self.num_embed(label)
      reconstruction = self.decoder(z, emb)
      return reconstruction, mu, logvar

  def train(self, dataloader, optimizer, recon_loss, kl_loss, alpha=1.0, beta=1.0, epochs=10, device=t.device('cpu')):
    self.train()
    for epoch in range(epochs):
      pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
      for spec, label in pbar:
        spec, label = spec.to(device), label.long().to(device)
        optimizer.zero_grad()

        x, mu, logvar = self(spec, label)

        recon = alpha * recon_loss(x, spec)
        kl = beta * kl_loss(mu, logvar)
        loss = recon + kl
        loss.backward()
        optimizer.step()
        pbar.set_postfix_str(f"recon_loss {recon.item():.4f}, kl_loss {kl.item():.4f}, total: {loss.item():.4f}")

      print(f"Epoch {epoch+1} Loss: {loss.item()}")

  @t.no_grad()
  def generate_sample(self, label=None, batch_size=4, device='cpu'):
    z = t.randn(batch_size, self.latent_dim, device=device)

    if label is None:
      label = t.randint(0, self.num_classes, (batch_size,), device=device)

    emb = self.num_embed(label)
    return self.decoder(z, emb)

  @t.no_grad()
  def visualize_sample(self, test_dataset, batch_size=4, device='cpu'):
    self.eval()
    sample, label = next(iter(DataLoader(test_dataset, batch_size=batch_size, shuffle=True)))

    assert callable(getattr(self, "generate_sample")), f"Class {__name__} should have generate_sample method implemented."
    gen = self.generate_sample(label=label, batch_size=batch_size, device=device).cpu().numpy()
    recon = self(sample, label)[0].cpu().numpy()
    sample = sample.cpu().numpy()
    #print("mean", gen.mean(), "std", gen.std(), "min", gen.min(), "max", gen.max())

    fig, axes = plt.subplots(batch_size, 3, figsize=(21, 15))
    axes = np.atleast_2d(axes)

    for i in range(batch_size):
        orig_spec = sample[i][0]
        gen_spec = gen[i][0]
        recon_spec = recon[i][0]

        specs_dict = dict(zip(["Original", "Generated", "Reconstructed"], [orig_spec, gen_spec, recon_spec]))

        for k, (title, spec) in enumerate(specs_dict.items()):
            ax = axes[i, k]
            img = librosa.display.specshow(
                spec,
                sr=22050,
                hop_length=256,
                x_axis='time',
                y_axis='mel',
                ax=ax
            )
            ax.set_title(f"{title} Spectrogram (Digit {label[i].item()})")
            fig.colorbar(img, format='%+2.0f dB', ax=ax)

    plt.tight_layout()
    plt.show()

    return orig_spec, gen_spec, recon_spec