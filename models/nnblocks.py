import torch as t
import torch.nn as nn

class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, time_dim, embed_dim):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, padding_mode='reflect')
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
    self.time_proj = nn.Linear(time_dim + embed_dim, out_channels)
    self.groupnorm = nn.GroupNorm(8, out_channels)
    self.gelu = nn.GELU()

  def forward(self, x, time_emb):
    x = self.conv1(x)
    x = self.gelu(x)
    x = self.conv2(x)
    x = self.groupnorm(x)
    t_emb = self.time_proj(time_emb)[:, :, None, None]
    return self.gelu(x + t_emb)


class TransConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, time_dim, embed_dim, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)):
    super().__init__()
    self.trans_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    self.time_proj = nn.Linear(time_dim + embed_dim, out_channels)
    self.groupnorm = nn.GroupNorm(8, out_channels)
    self.gelu = nn.GELU()

  def forward(self, x, time_emb):
    x = self.trans_conv(x)
    x = self.groupnorm(x)
    t_emb = self.time_proj(time_emb)[:, :, None, None]
    return self.gelu(x + t_emb)


class ResBlock(nn.Module):
  def __init__(self, channels, time_dim, embed_dim):
    super().__init__()
    self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect')
    self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect')
    self.time_proj = nn.Linear(time_dim + embed_dim, channels)
    self.groupnorm = nn.GroupNorm(8, channels)
    self.silu = nn.SiLU()

  def forward(self, x, time_emb):
    h = self.conv1(x)
    h = self.groupnorm(h)
    t_emb = self.time_proj(time_emb)[:, :, None, None]
    h = self.silu(h + t_emb)
    h = self.conv2(h)
    return self.silu(x + h)