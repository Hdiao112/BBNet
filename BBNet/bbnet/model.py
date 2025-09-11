import torch
import torch.nn as nn
import torch.nn.functional as F

class ResMLPWithAttn(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=4096, depth=8,
                 dropout_p=0.3, n_heads=8, out_dim=2):
        super().__init__()
        self.input_lin = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                          num_heads=n_heads,
                                          dropout=dropout_p,
                                          batch_first=True)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p),
            )
            for _ in range(depth)
        ])
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = F.gelu(self.input_lin(x))
        h_seq = h.unsqueeze(1)
        attn_out, _ = self.attn(h_seq, h_seq, h_seq)
        h = h + attn_out.squeeze(1)
        for blk in self.blocks:
            h = blk(h) + h
        return self.head(h)


class ResidualNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, depth=6, dropout_p=0.3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_p)]
        for _ in range(depth):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_p)]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
