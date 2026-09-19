import torch.nn as nn

class SelfAttentionLayer(nn.Module):
    def __init__(self, channels, time_dim, embed_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.time_proj = nn.Linear(time_dim + embed_dim, channels)
        self.groupnorm = nn.GroupNorm(8, num_channels=channels)
        self.flatten = nn.Flatten(start_dim=2, end_dim=-1)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)

    def forward(self, x, time_emb):
        B, C, H, W = x.shape
        t_emb = self.time_proj(time_emb)[:, :, None, None]
        h = x + t_emb

        h = self.groupnorm(h)
        qkv = self.flatten(h).permute(0, 2, 1)

        attn_out, _ = self.attn(qkv, qkv, qkv)

        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)

        return x + attn_out