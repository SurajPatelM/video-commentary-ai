"""
Audio Encoder module

Converts audio MFCC tensors shaped (B, n_mfcc, T) into fixed-size embeddings (B, D)
using a variety of aggregation/encoding techniques selected by name.

Usage
-----
from audio_encoder import AudioEncoder

enc = AudioEncoder(
    encoder_name="conv1d",   # one of: mean, mean_std, attn_pool, conv1d, conv1d_gem, cnn2d, gru_last, transformer_cls, stat_pool, flatten_adapt
    n_mfcc=64,
    projection_dim=256,
    freeze_backbone=False,    # optional: set True to freeze backbone like in the VisionEncoder example
    hidden_size=256           # technique-specific kwargs (see each encoder)
)

x = torch.randn(8, 64, 300)           # (B, n_mfcc, T)
lengths = torch.tensor([300]*8)       # optional valid lengths per sample
emb = enc(x, lengths=lengths)         # (B, projection_dim)

Design Notes
------------
- Mirrors the pattern of the provided VisionEncoder: a backbone (self.audio_model) and a projection layer.
- self.projection is nn.LazyLinear so you don't need to pre-compute the backbone output size.
- If freeze_backbone=True, the backbone is set to eval() and gradients are disabled.
- All encoders accept variable-length sequences via an optional `lengths` tensor (B,) when reasonable.

"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------
# Utility: masking helpers
# -------------------------------
def _make_time_mask(x: torch.Tensor, lengths: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Create a broadcastable mask for (B, C, T) given lengths (B,).
    Returns mask of shape (B, 1, T) with 1.0 for valid, 0.0 for pad; or None.
    """
    if lengths is None:
        return None
    B, _, T = x.shape
    device = x.device
    t = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
    mask = (t < lengths.view(-1, 1)).float()        # (B, T)
    return mask.unsqueeze(1)                        # (B, 1, T)


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], dim: int) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=dim)
    # mask expected to be broadcastable to x
    x_masked = x * mask
    denom = mask.sum(dim=dim).clamp_min(1e-6)
    return x_masked.sum(dim=dim) / denom


def _masked_std(x: torch.Tensor, mask: Optional[torch.Tensor], dim: int) -> torch.Tensor:
    mean = _masked_mean(x, mask, dim)
    if mask is None:
        var = ((x - mean.unsqueeze(dim)) ** 2).mean(dim=dim)
    else:
        var = (((x - mean.unsqueeze(dim)) ** 2) * mask).sum(dim=dim) / mask.sum(dim=dim).clamp_min(1e-6)
    return torch.sqrt(var.clamp_min(1e-6))


# -------------------------------
# Pooling blocks
# -------------------------------
class GeM(nn.Module):
    """Generalized Mean Pooling over the last dimension.
    y = (mean(x^p))^(1/p)
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps)
        x = x.pow(self.p)
        x = x.mean(dim=-1)
        return x.pow(1.0 / self.p)


# -------------------------------
# Backbones (all output (B, F_out))
# -------------------------------
class MeanPoolBackbone(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, C, T) => (B, C)
        mask = _make_time_mask(x, lengths)
        return _masked_mean(x, mask, dim=2)


class MeanStdPoolBackbone(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, C, T) => (B, 2C)
        mask = _make_time_mask(x, lengths)
        mean = _masked_mean(x, mask, dim=2)
        std  = _masked_std(x, mask, dim=2)
        return torch.cat([mean, std], dim=1)


class AttentionPoolBackbone(nn.Module):
    def __init__(self, in_dim: int, attn_hidden: int = 128):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, attn_hidden)
        self.lin2 = nn.Linear(attn_hidden, 1)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, C, T) => (B, C) via attention over T
        B, C, T = x.shape
        xT = x.transpose(1, 2)              # (B, T, C)
        h = torch.tanh(self.lin1(xT))       # (B, T, H)
        scores = self.lin2(h).squeeze(-1)   # (B, T)
        if lengths is not None:
            device = x.device
            t = torch.arange(T, device=device)[None, :]
            mask = (t >= lengths[:, None])  # True for pads
            scores = scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        pooled = (xT * weights).sum(dim=1)                    # (B, C)
        return pooled


class Conv1dBackbone(nn.Module):
    def __init__(self, in_ch: int, hidden_size: int = 256, num_layers: int = 2, kernel_size: int = 5):
        super().__init__()
        layers = []
        ch = in_ch
        pad = kernel_size // 2
        for i in range(num_layers):
            layers += [
                nn.Conv1d(ch, hidden_size, kernel_size=kernel_size, padding=pad),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
            ]
            ch = hidden_size
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.net(x)                 # (B, H, T)
        return h.mean(dim=2)            # global avg pool over time -> (B, H)


class Conv1dGeMBackbone(nn.Module):
    def __init__(self, in_ch: int, hidden_size: int = 256, num_layers: int = 2, kernel_size: int = 5, p: float = 3.0):
        super().__init__()
        self.conv = Conv1dBackbone(in_ch, hidden_size, num_layers, kernel_size)
        self.gem = GeM(p=p)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.conv.net(x)   # (B, H, T)
        return self.gem(h)     # (B, H)


class CNN2DBackbone(nn.Module):
    def __init__(self, channels: tuple[int, int, int] = (32, 64, 128), kernel_size: int = 3):
        """2D CNN over the (n_mfcc x T) map.
        Note: input is treated as a single-channel image with height=n_mfcc and width=T.
        """
        super().__init__()
        k = kernel_size
        p = k // 2
        c1, c2, c3 = channels
        # in_channels is 1 because we add a singleton channel dim in forward()
        self.net = nn.Sequential(
            nn.Conv2d(1, c1, (k, k), padding=(p, p)), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, (k, k), padding=(p, p)), nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, (k, k), padding=(p, p)), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, C, T) -> treat as (B, 1, C, T)
        inp = x.unsqueeze(1)                 # (B, 1, n_mfcc, T)
        feat = self.net(inp)                 # (B, C3, 1, 1)
        return feat.flatten(1)               # (B, C3)


class GRUBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int = 256, num_layers: int = 1, bidirectional: bool = True, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, C, T) -> (B, T, C)
        xT = x.transpose(1, 2)
        if lengths is not None:
            # pack padded sequence
            packed = nn.utils.rnn.pack_padded_sequence(xT, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, h = self.gru(packed)
            # h: (num_layers * num_directions, B, hidden_size)
        else:
            out, h = self.gru(xT)
        if self.bidirectional:
            # Concatenate last layer's forward and backward hidden
            h_f = h[-2, :, :]
            h_b = h[-1, :, :]
            last = torch.cat([h_f, h_b], dim=1)  # (B, 2H)
        else:
            last = h[-1, :, :]                   # (B, H)
        return last


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        T = x.size(1)
        return x + self.pe[:T, :].unsqueeze(0)


class TransformerCLSBackbone(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 256, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, C, T) -> (B, T, C)
        xT = x.transpose(1, 2)
        z = self.proj_in(xT)
        z = self.pos(z)
        B = z.size(0)
        cls_tok = self.cls.expand(B, -1, -1)   # (B, 1, d_model)
        z = torch.cat([cls_tok, z], dim=1)     # (B, 1+T, d_model)
        # key padding mask: True for PAD positions
        key_pad_mask = None
        if lengths is not None:
            T = x.size(2)
            device = x.device
            t = torch.arange(T, device=device)[None, :]
            pad = (t >= lengths[:, None])  # (B, T)
            # prepend False for CLS (never padded)
            key_pad_mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=device), pad], dim=1)
        out = self.encoder(z, src_key_padding_mask=key_pad_mask)  # (B, 1+T, d_model)
        cls = out[:, 0, :]                                        # (B, d_model)
        return cls


class StatPoolBackbone(nn.Module):
    """Compute statistical functionals over time per MFCC channel and concatenate."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, C, T)
        mask = _make_time_mask(x, lengths)
        mean = _masked_mean(x, mask, dim=2)
        std = _masked_std(x, mask, dim=2)
        # Quantiles (try torch.quantile, fallback to approx using kthvalue)
        B, C, T = x.shape
        if lengths is None:
            try:
                q = torch.quantile(x, torch.tensor([0.25, 0.5, 0.75], device=x.device), dim=2)
                q25, q50, q75 = q[0], q[1], q[2]
            except Exception:
                # fallback approx
                k25 = max(1, int(0.25 * T))
                k50 = max(1, int(0.50 * T))
                k75 = max(1, int(0.75 * T))
                q25 = x.kthvalue(k25, dim=2).values
                q50 = x.kthvalue(k50, dim=2).values
                q75 = x.kthvalue(k75, dim=2).values
        else:
            # masked quantile: approximate by sorting valid frames
            q25_list, q50_list, q75_list = [], [], []
            for b in range(B):
                t_valid = int(lengths[b].item())
                xb = x[b, :, :t_valid]
                try:
                    q = torch.quantile(xb, torch.tensor([0.25, 0.5, 0.75], device=x.device), dim=1)
                    q25_list.append(q[0])
                    q50_list.append(q[1])
                    q75_list.append(q[2])
                except Exception:
                    k25 = max(1, int(0.25 * t_valid))
                    k50 = max(1, int(0.50 * t_valid))
                    k75 = max(1, int(0.75 * t_valid))
                    q25_list.append(xb.kthvalue(k25, dim=1).values)
                    q50_list.append(xb.kthvalue(k50, dim=1).values)
                    q75_list.append(xb.kthvalue(k75, dim=1).values)
            q25 = torch.stack(q25_list, dim=0)
            q50 = torch.stack(q50_list, dim=0)
            q75 = torch.stack(q75_list, dim=0)
        return torch.cat([mean, std, q25, q50, q75], dim=1)


class FlattenAdaptiveBackbone(nn.Module):
    """Adaptively average-pool time to a fixed length L, then flatten (B, C, L).
    This mimics FLATTEN while remaining robust to variable-length inputs.
    """
    def __init__(self, target_len: int = 8):
        super().__init__()
        assert target_len > 0
        self.pool = nn.AdaptiveAvgPool1d(target_len)
        self.target_len = target_len

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.pool(x)                # (B, C, L)
        return y.flatten(start_dim=1)   # (B, C*L)


# ---------------------------------
# Registry of available backbones
# ---------------------------------
AUDIO_ENCODERS = {
    'mean': MeanPoolBackbone,                   # (B, C)
    'mean_std': MeanStdPoolBackbone,            # (B, 2C)
    'attn_pool': AttentionPoolBackbone,         # (B, C)
    'conv1d': Conv1dBackbone,                   # (B, H)
    'conv1d_gem': Conv1dGeMBackbone,            # (B, H)
    'cnn2d': CNN2DBackbone,                     # (B, C3)
    'gru_last': GRUBackbone,                    # (B, 2H if bidir else H)
    'transformer_cls': TransformerCLSBackbone,  # (B, d_model)
    'stat_pool': StatPoolBackbone,              # (B, 5C)
    'flatten_adapt': FlattenAdaptiveBackbone,   # (B, C*L)
}


# -------------------------------
# Public AudioEncoder (like VisionEncoder)
# -------------------------------
class AudioEncoder(nn.Module):
    """Audio encoder wrapper with selectable backbone and projection layer.

    Parameters
    ----------
    encoder_name : str
        One of AUDIO_ENCODERS keys.
    n_mfcc : int
        Number of MFCC channels in the input.
    projection_dim : int
        Output embedding dimensionality D.
    freeze_backbone : bool, optional
        If True, run backbone in eval mode and disable its gradients.
    **kwargs : dict
        Technique-specific kwargs forwarded to the backbone constructor, e.g.:
        - AttentionPoolBackbone: attn_hidden
        - Conv1dBackbone/Conv1dGeMBackbone: hidden_size, num_layers, kernel_size, p
        - CNN2DBackbone: channels, kernel_size
        - GRUBackbone: hidden_size, num_layers, bidirectional, dropout
        - TransformerCLSBackbone: d_model, nhead, num_layers, dim_feedforward, dropout
        - FlattenAdaptiveBackbone: target_len

    Notes
    -----
    - Input x must be shaped (B, n_mfcc, T).
    - Optional `lengths` (B,) can be provided for masked/packed operations.
    - Projection is nn.LazyLinear to adapt to backbone output size automatically.
    """

    def __init__(self, encoder_name: str, n_mfcc: int, projection_dim: int, freeze_backbone: bool = False, **kwargs):
        super().__init__()
        if encoder_name not in AUDIO_ENCODERS:
            raise ValueError(f"Unknown encoder_name '{encoder_name}'. Available: {list(AUDIO_ENCODERS.keys())}")

        # Instantiate backbone; pass in_dim where needed
        Backbone = AUDIO_ENCODERS[encoder_name]
        try:
            # Backbones that require in_dim/n_mfcc
            self.audio_model = Backbone(in_dim=n_mfcc, **kwargs)
        except TypeError:
            try:
                self.audio_model = Backbone(in_ch=n_mfcc, **kwargs)
            except TypeError:
                # Backbones that take no dim args
                self.audio_model = Backbone(**kwargs)

        self.projection = nn.LazyLinear(projection_dim)

        if freeze_backbone:
            for p in self.audio_model.parameters():
                p.requires_grad = False
            self.audio_model.eval()
        else:
            self.audio_model.train()

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input MFCCs, shape (B, n_mfcc, T).
        lengths : Optional[torch.Tensor]
            Valid lengths per batch element, shape (B,). Optional.
        """
        features = self.audio_model(x, lengths=lengths)
        projected = self.projection(features)
        return projected


# -------------------------------
# Small self-test (run manually)
# -------------------------------
if __name__ == "__main__":
    B, C, T = 4, 40, 321
    x = torch.randn(B, C, T)
    lengths = torch.tensor([321, 300, 250, 321])

    for name, kwargs in [
        ("mean", {}),
        ("mean_std", {}),
        ("attn_pool", {"attn_hidden": 64}),
        ("conv1d", {"hidden_size": 128, "num_layers": 2}),
        ("conv1d_gem", {"hidden_size": 128, "num_layers": 2, "p": 3.0}),
        ("cnn2d", {"channels": (16, 32, 64)}),
        ("gru_last", {"hidden_size": 64, "num_layers": 1, "bidirectional": True}),
        ("transformer_cls", {"d_model": 128, "nhead": 4, "num_layers": 2}),
        ("stat_pool", {}),
        ("flatten_adapt", {"target_len": 8}),
    ]:
        print(f"Testing {name} ...", end=" ")
        enc = AudioEncoder(name, n_mfcc=C, projection_dim=96, **kwargs)
        y = enc(x, lengths=lengths)
        assert y.shape == (B, 96), f"{name} produced shape {y.shape}"
        print("ok")
