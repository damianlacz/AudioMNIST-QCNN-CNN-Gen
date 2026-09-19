import torch as t
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import librosa

from torch.utils.data import DataLoader
from models.nnblocks import ConvBlock, TransConvBlock, ResBlock
from models.attn import SelfAttentionLayer
from tqdm import tqdm

class ForwardDiffusionProcess:
  def __init__(self, start=1e-4, end=0.02, n_steps=1000):
    self.start = start
    self.end = end
    self.n_steps = n_steps
    self.betas = self._beta_scheduler(start, end, n_steps)
    self.alphas = 1.0 - self.betas
    self.alphas_bar = t.cumprod(self.alphas, dim=0) # fixed axis=0 to dim=0

  def _beta_scheduler(self, start=1e-4, end=0.02, n_steps=1000):
    return t.linspace(start, end, n_steps)

  def calc_noisy_image(self, x_0, time):
    device = x_0.device
    alpha_bar_t = self.alphas_bar[time.cpu()].view(-1, 1, 1, 1).to(device)
    eps = t.randn_like(x_0)
    return t.sqrt(alpha_bar_t) * x_0 + t.sqrt(1.0 - alpha_bar_t) * eps, eps

class TimeEmbedding(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim
    self.mlp = nn.Sequential(
        nn.Linear(dim, dim * 4),
        nn.SiLU(),
        nn.Linear(dim * 4, dim)
    )

  def forward(self, time):
    half_dim = self.dim // 2
    freq = t.exp(-0.5 * np.log(10000) * t.arange(start=0, end=half_dim, dtype=t.float32, device=time.device) / half_dim)
    emb = time[:, None] * freq[None, :]
    emb = t.cat((emb.sin(), emb.cos()), dim=-1)
    return self.mlp(emb)

class InverseDiffusionProcess(nn.Module):
    def __init__(self, time_dim=64, embed_dim=64):
        super(InverseDiffusionProcess, self).__init__()
        self.time_dim = time_dim
        self.embed_dim = embed_dim

        self.time_emb = TimeEmbedding(time_dim)
        self.num_emb = nn.Embedding(10, embed_dim)

        # Encoder Path
        self.enc_block1 = ConvBlock(1, 16, time_dim=time_dim, embed_dim=embed_dim)

        self.enc_block2 = ConvBlock(16, 32, time_dim=time_dim, embed_dim=embed_dim)
        #self.attn1 = SelfAttentionLayer(channels=32, time_dim=time_dim, embed_dim=embed_dim, num_heads=4)

        self.enc_block3 = ConvBlock(32, 64, time_dim=time_dim, embed_dim=embed_dim)
        self.attn2 = SelfAttentionLayer(channels=64, time_dim=time_dim, embed_dim=embed_dim, num_heads=4)

        self.res_block = ResBlock(64, time_dim=time_dim, embed_dim=embed_dim)
        self.res_attn = SelfAttentionLayer(channels=64, time_dim=time_dim, embed_dim=embed_dim, num_heads=4)

        self.dec_block1 = TransConvBlock(128, 32, time_dim=time_dim, embed_dim=embed_dim, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.dec_attn1 = SelfAttentionLayer(channels=32, time_dim=time_dim, embed_dim=embed_dim, num_heads=4)

        self.dec_block2 = TransConvBlock(64, 16, time_dim=time_dim, embed_dim=embed_dim, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        #self.dec_attn2 = SelfAttentionLayer(channels=16, time_dim=time_dim, embed_dim=embed_dim, num_heads=4)

        # Final output conv to restore exact 80x87 resolution
        self.dec_conv3 = nn.ConvTranspose2d(32, 1, kernel_size=(4, 3), stride=(2, 2), padding=(1, 1))

    def forward(self, x, time, label):
        t_emb = self.time_emb(time)
        num_emb = self.num_emb(label)
        t_emb = t.cat([t_emb, num_emb], dim=1)

        e1 = self.enc_block1(x, t_emb)    # Shape: [B, 16, 40, 44]

        e2 = self.enc_block2(e1, t_emb)   # Shape: [B, 32, 20, 22]
        #e2 = self.attn1(e2, t_emb)

        e3 = self.enc_block3(e2, t_emb)   # Shape: [B, 64, 10, 11]
        e3 = self.attn2(e3, t_emb)

        x = self.res_block(e3, t_emb)      # Shape: [B, 64, 10, 11]
        x = self.res_attn(x, t_emb)

        x = t.cat([x, e3], dim=1)          # Channels: 64 + 64 = 128
        x = self.dec_block1(x, t_emb)      # Out: 32 channels, shape: [B, 32, 20, 22]
        x = self.dec_attn1(x, t_emb)

        x = t.cat([x, e2], dim=1)          # Channels: 32 + 32 = 64
        x = self.dec_block2(x, t_emb)      # Out: 16 channels, shape: [B, 16, 40, 44]
        #x = self.dec_attn2(x, t_emb)

        x = t.cat([x, e1], dim=1)          # Channels: 16 + 16 = 32
        return self.dec_conv3(x)           # Out: 1 channel, shape: [B, 1, 80, 87]

class DiffusionModel(nn.Module):
  def __init__(self, start=1e-4, end=0.02, n_steps=1000, time_dim=64, embed_dim=64):
    super(DiffusionModel, self).__init__()
    self.fwdprocess = ForwardDiffusionProcess(start=start, end=end, n_steps=n_steps)
    self.model = InverseDiffusionProcess(time_dim=time_dim, embed_dim=embed_dim)
    self.betas = self.fwdprocess.betas
    self.alphas = self.fwdprocess.alphas
    self.alphas_bar = self.fwdprocess.alphas_bar
    self.n_steps = n_steps

  def forward(self, x, time, label):
    return self.model(x, time, label)

  def train(self, dataloader, optimizer, loss_fn, alpha=100.0, epochs=10, device=t.device('cpu')):
    self.train()
    for epoch in range(epochs):
      pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
      for x_0, label in pbar:
        x_0, label = x_0.to(device), label.long().to(device)

        optimizer.zero_grad()

        time = t.randint(0, self.n_steps, (x_0.shape[0],), device=device)
        x_t, noise = self.fwdprocess.calc_noisy_image(x_0, time)
        pred_noise = self(x_t, time, label)
        loss = alpha * loss_fn(pred_noise, noise)

        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.item())
      print(f"Epoch {epoch+1} Loss: {loss.item()}")

  @t.no_grad()
  def _reverse_step(self, x_t, time, label):
    self.model.eval()
    device = x_t.device
    time, label = time.to(device), label.to(device)
    beta_t = self.betas[time.cpu()][:, None, None, None].to(device)
    alpha_t = self.alphas[time.cpu()][:, None, None, None].to(device)
    alpha_bar_t = self.alphas_bar[time.cpu()][:, None, None, None].to(device)
    mean = (1.0 / t.sqrt(alpha_t)) * (x_t - beta_t / t.sqrt(1.0 - alpha_bar_t) * self.model(x_t, time, label))
    if t.all(time == 0):
        return mean

    prev_time = t.clamp(time - 1, 0)
    prev_alpha_bar_t = self.alphas_bar[prev_time][:, None, None, None].to(device)
    posterior_var = beta_t * (1.0 - prev_alpha_bar_t) / (1.0 - alpha_bar_t)
    eps = t.randn_like(x_t)
    return mean + t.sqrt(posterior_var) * eps

  @t.no_grad()
  def _reconstruct_spec(self, x_t, label=None, batch_size=4, device=t.device('cpu')):
    for step in reversed(range(self.n_steps)):
      time = t.full((batch_size,), step, dtype=t.long, device=device)
      x_t = self._reverse_step(x_t, time, label)
    return x_t

  @t.no_grad()
  def generate_sample(self, label=None, batch_size=4, device=t.device('cpu')):
    x = t.randn(batch_size, 1, 80, 87, device=device)
    label = t.randint(0, 10, (batch_size,), dtype=t.long, device=device) if label is None else label.long().to(device)
    x = self._reconstruct_spec(x, label=label, batch_size=batch_size, device=device)
    return x

  @t.no_grad()
  def visualize_sample(self, test_dataset, batch_size=4, device=t.device('cpu')):
    sample, label = next(iter(DataLoader(test_dataset, batch_size=batch_size, shuffle=True)))

    assert callable(getattr(self, "generate_sample")), f"Class {__name__} should have generate_sample method implemented."
    gen = self.generate_sample(label=label, batch_size=batch_size, device=device).cpu().numpy()
    noisy_orig = self.fwdprocess.calc_noisy_image(sample, t.tensor([self.n_steps - 1]))[0]
    recon = self._reconstruct_spec(noisy_orig, label).cpu().numpy()
    sample = sample.cpu().numpy()

    fig, axes = plt.subplots(batch_size, 3, figsize=(21, 15))
    axes = np.atleast_2d(axes)

    for i in range(batch_size):
      orig_spec = sample[i][0]
      gen_spec = gen[i][0]
      recon_spec = recon[i][0]

      specs_dict = dict(zip(["Original", "Generated", "Reconstructed"], [orig_spec, gen_spec, recon_spec]))

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

    return orig_spec, gen_spec, recon_spec
