import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class Swish(nn.Module):
    def forward(self, x):
        return F.silu(x)


class ConformerConvModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pw_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.dw_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.swish = Swish()
        self.pw_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layer_norm(x)
        y = y.transpose(1, 2)
        y = self.pw_conv1(y)
        y = self.glu(y)
        y = self.dw_conv(y)
        y = self.bn(y)
        y = self.swish(y)
        y = self.pw_conv2(y)
        y = y.transpose(1, 2)
        return self.dropout(y)


class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.swish = Swish()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layer_norm(x)
        y = self.linear1(y)
        y = self.swish(y)
        y = self.dropout(y)
        y = self.linear2(y)
        y = self.dropout2(y)
        return y


class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1), :].unsqueeze(0)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        y = self.layer_norm(x)
        y, _ = self.attn(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
        return self.dropout(y)


class ConformerBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, conv_kernel: int, dropout: float
    ):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, d_ff, dropout)
        self.mhsa = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.conv = ConformerConvModule(d_model, conv_kernel, dropout)
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)
        self.final_ln = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.mhsa(x, key_padding_mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        x = self.final_ln(x)
        return x


class ConvSubsampling(nn.Module):
    def __init__(self, in_channels: int, d_model: int, n_feats: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, d_model // 2, kernel_size=3, stride=(1, 2), padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(d_model // 2, d_model, kernel_size=3, stride=(1, 2), padding=1),
            nn.ReLU(),
        )
        self.n_feats = n_feats
        self.out_dim = d_model

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        x = x.unsqueeze(1)
        x = self.conv(x)
        B, D, F, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, D * F)
        return x, self._downsample_lengths(lengths)

    def _downsample_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        return torch.div(
            torch.div(lengths, 2, rounding_mode="floor"), 2, rounding_mode="floor"
        )


class LinearProjection(nn.Module):
    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)

    def forward(self, x):
        return self.proj(x)


class ConformerModel(nn.Module):
    def __init__(
        self,
        n_feats: int,
        n_tokens: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 16,
        ff_multiplier: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_tokens = n_tokens

        self.subsample = ConvSubsampling(
            in_channels=1, d_model=d_model, n_feats=n_feats
        )
        self.pre_proj = LinearProjection(in_dim=d_model * n_feats, d_model=d_model)

        self.pos_enc = RelativePositionalEncoding(d_model)

        self.blocks = nn.ModuleList(
            [
                ConformerBlock(d_model, n_heads, ff_multiplier * d_model, conv_kernel, dropout)
                for _ in range(num_layers)
            ]
        )

        self.ctc_head = nn.Linear(d_model, n_tokens)

    def forward(
        self, spectrogram: torch.Tensor, spectrogram_length: torch.Tensor, **batch
    ):
        x, out_lengths = self.subsample(spectrogram, spectrogram_length)
        x = self.pre_proj(x)

        x = self.pos_enc(x)
        B, T, _ = x.shape
        device = x.device
        time_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        key_padding_mask = time_ids >= out_lengths.to(device).unsqueeze(1)

        for blk in self.blocks:
            x = blk(x, key_padding_mask)

        logits = self.ctc_head(x)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs_length = self.transform_input_lengths(spectrogram_length)

        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        return torch.div(
            torch.div(input_lengths, 2, rounding_mode="floor"), 2, rounding_mode="floor"
        )

    def __str__(self):
        all_parameters = sum(p.numel() for p in self.parameters())
        trainable_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        result_info = super().__str__()
        result_info += f"\nAll parameters: {all_parameters}"
        result_info += f"\nTrainable parameters: {trainable_parameters}"
        return result_info
