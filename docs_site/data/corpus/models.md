# Model Architecture Documentation

This file documents the core model architecture from src/models.py for the documentation site.

## Overview

The Latent-Space-Model architecture consists of four main components:

1. LatentEncoder: ModernBERT + pooling + projection to latent sequence
2. MiddleTransformer: Transformer for latent reasoning over sequences
3. PrefixAdapter: Cross-attention from latent sequence to prefix embeddings
4. LatentSpaceModel: Combined full model

## Core Components

### RMSNorm

Root Mean Square Layer Normalization used throughout the architecture.

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x_norm = x / rms
        return x_norm * self.weight
```

### SwiGLU Activation

SwiGLU (SiLU-gated linear unit) activation function used in feed-forward networks.

```python
class SwiGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.proj(x)
        x_gated, x_linear = x_proj.chunk(2, dim=-1)
        return F.silu(x_gated) * x_linear
```

### Rotary Embedding (RoPE)

Rotary positional embeddings for position-aware attention.

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
```

## LatentEncoder

The encoder uses ModernBERT to convert text tokens to a compact latent sequence.

```python
class LatentEncoder(nn.Module):
    def __init__(self, model_name: str, hidden_dim: int = 768, latent_dim: int = 256,
                 latent_seq_len: int = 8, num_unfrozen_layers: int = 0, ...):
```

The encoder performs:
1. ModernBERT encoding: [B, L, 768]
2. Mean pooling to [B, 768]
3. Compression MLP to [B, 8, 256]

## MultiHeadSelfAttention

Self-attention mechanism with optional RoPE.

Configuration:
- dim: 256 (latent dimension)
- num_heads: 4
- head_dim: 64 (256 / 4)
- use_rope: True

The attention pattern across 8 latent positions allows each vector to attend to the others.

## TransformerBlock

Core building block of the MiddleTransformer.

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_multiplier: float = 4.0, ...):
        self.norm1 = RMSNorm(dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(dim, num_heads, use_rope=True)
        self.norm2 = RMSNorm(dim, eps=1e-6)
        self.ffn = nn.Sequential(
            SwiGLU(dim, hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

## MiddleTransformer

The core reasoning module operating on latent sequences.

```python
class MiddleTransformer(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_multiplier: float = 4.0,
        dropout: float = 0.1,
        use_rope: bool = True,
        rope_base: int = 10000
    )
```

Tensor shapes:
- Input: [B, 8, 256]
- Output: [B, 8, 256]
- Internal FFN: 256 * 4 = 1024 hidden units

## PrefixAdapter

Cross-attention bridge between latent space and decoder embeddings.

```python
class PrefixAdapter(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        decoder_dim: int = 640,
        prefix_len: int = 50,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_rope: bool = True
    )
```

Cross-attention flow:
- Query: [B, 50, 640] (learnable positions)
- Key/Value: [B, 8, 256] (from latent)
- Output: [B, 50, 640] (prefix embeddings)

## LatentSpaceModel

Full model combining all components.

Data flow:
1. Encode: text -> ModernBERT -> [B, 8, 256]
2. Middle: [B, 8, 256] -> TransformerBlock x4 -> [B, 8, 256]
3. Adapt: latent -> CrossAttention -> [B, 50, 640]
4. Decode: prefix -> Gemma -> generated text

## Stop Latent Mechanism

The model supports a learnable stop latent vector for generation control.

```python
def _init_stop_latent(latent_seq_len: int, latent_dim: int,
                      init: str = "random_normalized",
                      seed: int = 1337) -> torch.Tensor
```

Initialization options:
- zero: All zeros
- random_normalized: Random unit vectors

The stop latent is:
- Learned during training
- Appended to batches for augmentation
- Can signal end-of-generation

## Complete Data Flow

```
Input Text (Dialogue)
    |
ModernBERT Encoder       [B, L, 768]
    |
Mean Pooling             [B, 768]
    |
Compression MLP          [B, 8, 256] (z_in)
    |
MiddleTransformer        [B, 8, 256] (z_out)
    |      (4 layers, self-attention)
    |
Prefix Adapter (Cross)   [B, 50, 640]
    |
Gemma-3-270m Decoder
    |
Output Text (Summary)
```

## Key Design Decisions

1. Multiple latent vectors (8) instead of single bottleneck
2. Self-attention across latent positions for information exchange
3. 4 transformer layers for multi-step reasoning
4. Cross-attention for latent-to-decoder mapping
5. RoPE for position-aware latent transformation
6. RMSNorm and SwiGLU for modern transformer architecture
