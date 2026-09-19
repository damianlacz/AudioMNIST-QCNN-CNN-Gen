import torch as t
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import librosa

from torch.utils.data import DataLoader
from tqdm import tqdm

class Generator(nn.Module):
  def __init__(self, noise_dim=16, embed_dim=64):
    super(Generator, self).__init__()
    self.num_emb = nn.Embedding(10, embedding_dim=embed_dim)
    self.input_layer = nn.Linear(noise_dim + embed_dim, 64 * 10 * 11)
    self.trans_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    #nn.init.uniform_(self.trans_conv1.weight)
    self.trans_conv2 = nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=(2, 2))
    #nn.init.uniform_(self.trans_conv2.weight)
    self.trans_conv3 = nn.ConvTranspose2d(16, 1, kernel_size=(4, 3), stride=(2, 2))
    #nn.init.uniform_(self.trans_conv3.weight)
    self.leaky_relu = nn.LeakyReLU(negative_slope=0.5)
    self.dropout = nn.Dropout(p=0.25)
    self.relu = nn.ReLU()

  def forward(self, x, label):
    label_emb = self.num_emb(label)
    x = t.cat([x, label_emb], dim=1)
    x = self.input_layer(x).view(-1, 64, 10, 11)
    x = self.leaky_relu(x)
    x = self.dropout(x)
    x = self.trans_conv1(x)
    x = self.leaky_relu(x)
    x = self.trans_conv2(x)
    x = self.leaky_relu(x)
    x = self.trans_conv3(x)
    return self.relu(x)

class Discriminator(nn.Module):
  def __init__(self, embed_dim=64):
    super(Discriminator, self).__init__()
    self.embed_dim = embed_dim
    self.num_emb = nn.Embedding(10, embedding_dim=embed_dim)
    self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
    self.dense = nn.Linear(128 * 10 * 11 + embed_dim, 512)
    self.out = nn.Linear(512, 1)

    self.relu = nn.ReLU()

  def forward(self, x, label):

    x = self.conv1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.relu(x)
    x = self.conv3(x)
    x = self.relu(x)
    x = self.flatten(x)

    emb = self.num_emb(label)
    x = t.cat([x, emb], dim=1)
    x = self.dense(x)
    x = self.out(x)
    return x

class GAN(nn.Module):
  def __init__(self, noise_dim=16, gen_embed_dim=64, disc_embed_dim=64, gen_lr=1e-4, disc_lr=1e-3):
    super(GAN, self).__init__()
    self.noise_dim = noise_dim
    self.gen_embed_dim = gen_embed_dim
    self.discriminator_embed_dim = disc_embed_dim
    self.generator = Generator(noise_dim=noise_dim, embed_dim=gen_embed_dim)
    self.discriminator = Discriminator(embed_dim=disc_embed_dim)

  def forward(self, x, label):
    noise = t.randn(x.shape[0], self.noise_dim)
    gen_spec = self.generator(noise, label)
    gen_prob = self.discriminator(gen_spec, label)
    spec_prob = self.discriminator(x, label)
    return gen_spec, gen_prob, spec_prob

  def train(self, dataloader, gen_opt, disc_opt, gen_loss_f, disc_loss_f, alpha=1.0, beta=1.0, epochs=10, device=t.device('cpu')):
    self.train()
    for epoch in range(epochs):
      pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
      for spec, label in pbar:
        spec, label = spec.to(device), label.to(device)
        gen_opt.zero_grad(set_to_none=True)

        batch_size = spec.shape[0]
        noise = t.randn(batch_size, self.noise_dim, device=device)
        gen_spec = self.generator(noise, label)

        gen_loss = alpha * gen_loss_f(spec, gen_spec)
        gen_loss.backward()
        gen_opt.step()

        disc_opt.zero_grad(set_to_none=True)
        gen_prob = self.discriminator(gen_spec.detach(), label)
        spec_prob = self.discriminator(spec, label)

        fake_loss = disc_loss_f(gen_prob, t.zeros_like(gen_prob, device=device))
        real_loss = disc_loss_f(spec_prob, t.ones_like(spec_prob, device=device))
        disc_loss = beta * (fake_loss + real_loss)

        disc_loss.backward()
        disc_opt.step()

        loss = gen_loss + disc_loss

        pbar.set_postfix(gen_loss=gen_loss.item(), fake_loss=fake_loss.item(), real_loss=real_loss.item(), disc_loss=disc_loss.item(), loss=loss.item())
      print(f"Epoch {epoch+1} Loss: {loss.item()}")

  @t.no_grad()
  def generate_sample(self, label=None, batch_size=4, device=t.device('cpu')) -> t.Tensor:
    if label is None:
      label = t.randint(0, 10, (batch_size,), dtype=t.long, device=device)

    noise = t.randn(batch_size, self.noise_dim, device=device)
    return self.generator(noise, label)

  @t.no_grad()
  def visualize_sample(self, test_dataloader, batch_size=4, device=t.device('cpu')) -> tuple[np.ndarray, np.ndarray]:
    sample, label = next(iter(DataLoader(test_dataloader, batch_size=batch_size, shuffle=True)))

    gen = self.generate_sample(label=label, batch_size=batch_size, device=device).cpu().numpy()
    sample = sample.cpu().numpy()

    fig, axes = plt.subplots(batch_size, 2, figsize=(21, 15))
    axes = np.atleast_2d(axes)

    for i in range(batch_size):
      orig_spec = sample[i][0]
      gen_spec = gen[i][0]

      specs_dict = dict(zip(["Original", "Generated"], [orig_spec, gen_spec]))
      for k, (title, spec_data) in enumerate(specs_dict.items()):
        ax = axes[i, k]
        img = librosa.display.specshow(
            spec_data,
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

    return orig_spec, gen_spec
