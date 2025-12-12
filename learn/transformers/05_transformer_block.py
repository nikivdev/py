#!/usr/bin/env python3
"""
Step 5: Transformer Block with Residual Connections & Layer Normalization

The engineering secret sauce that allows Transformers to be DEEP.

After each sub-layer (Attention or FFN):
  Output = LayerNorm(x + Sublayer(x))

- "+x" (Residual): Gradient highway, keeps original information
- LayerNorm: Stabilizes training, keeps numbers from exploding

Run: uv run python 05_transformer_block.py
"""

import mlx.core as mx
import mlx.nn as nn
import math


# ============================================================
# Components (from previous files)
# ============================================================


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    scores = mx.matmul(Q, K.swapaxes(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = mx.where(mask, scores, mx.array(float("-inf")))
    weights = mx.softmax(scores, axis=-1)
    return mx.matmul(weights, V), weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 0, 2)

    def combine_heads(self, x):
        x = x.transpose(1, 0, 2)
        seq_len = x.shape[0]
        return x.reshape(seq_len, self.d_model)

    def __call__(self, x, mask=None):
        Q = self.split_heads(self.W_Q(x))
        K = self.split_heads(self.W_K(x))
        V = self.split_heads(self.W_V(x))

        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)
        combined = self.combine_heads(attn_output)
        return self.dropout(self.W_O(combined))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        return self.dropout(self.W2(nn.relu(self.W1(x))))


# ============================================================
# The Transformer Block
# ============================================================


class TransformerBlock(nn.Module):
    """
    One complete Transformer encoder block.

    Structure (Post-LN, original paper):
        x → MultiHeadAttention → Add & LayerNorm → FFN → Add & LayerNorm → output

    Actually implemented as:
        x_attn = LayerNorm(x + MultiHeadAttention(x))
        output = LayerNorm(x_attn + FFN(x_attn))

    Modern variant (Pre-LN, more stable):
        x_attn = x + MultiHeadAttention(LayerNorm(x))
        output = x_attn + FFN(LayerNorm(x_attn))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        pre_norm: bool = True,  # Modern default
    ):
        super().__init__()

        self.pre_norm = pre_norm

        # Sub-layers
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)

    def __call__(
        self, x: mx.array, mask: mx.array | None = None, verbose: bool = False
    ) -> mx.array:
        if self.pre_norm:
            return self._forward_pre_norm(x, mask, verbose)
        else:
            return self._forward_post_norm(x, mask, verbose)

    def _forward_pre_norm(self, x, mask, verbose):
        """Pre-LN: Normalize before each sub-layer (more stable training)."""
        if verbose:
            print("\n  Using Pre-LN (modern, stable)")
            print(f"  Input: {x.shape}, mean={mx.mean(x).item():.4f}, std={mx.std(x).item():.4f}")

        # Attention sub-layer with residual
        normed = self.norm1(x)
        if verbose:
            print(f"  After LayerNorm1: mean={mx.mean(normed).item():.4f}, std={mx.std(normed).item():.4f}")

        attn_out = self.attention(normed, mask)
        x = x + attn_out  # Residual connection!
        if verbose:
            print(f"  After Attention + Residual: mean={mx.mean(x).item():.4f}")

        # FFN sub-layer with residual
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out  # Residual connection!
        if verbose:
            print(f"  After FFN + Residual: mean={mx.mean(x).item():.4f}")

        return x

    def _forward_post_norm(self, x, mask, verbose):
        """Post-LN: Normalize after each sub-layer (original paper)."""
        if verbose:
            print("\n  Using Post-LN (original paper)")

        # Attention sub-layer
        attn_out = self.attention(x, mask)
        x = self.norm1(x + attn_out)  # Add then norm

        # FFN sub-layer
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)  # Add then norm

        return x


def demo():
    print("=" * 60)
    print("TRANSFORMER BLOCK: RESIDUALS & LAYER NORM")
    print("=" * 60)

    d_model = 64
    num_heads = 8
    seq_len = 5

    mx.random.seed(42)
    x = mx.random.normal((seq_len, d_model))

    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_ff: {4 * d_model} (default 4x)")

    # Create transformer block
    block = TransformerBlock(d_model, num_heads, dropout=0.0)

    print("\n" + "-" * 50)
    print("TRANSFORMER BLOCK FORWARD PASS")
    print("-" * 50)

    output = block(x, verbose=True)

    print(f"\n  Final output shape: {output.shape}")

    print("\n" + "-" * 50)
    print("WHY RESIDUAL CONNECTIONS?")
    print("-" * 50)

    print("""
The Problem: Deep networks suffer from vanishing gradients.
  - Gradients multiply through each layer
  - With depth, gradients → 0 or → ∞
  - Network can't learn!

The Solution: Residual Connections (Skip Connections)

  output = x + Sublayer(x)

Instead of learning H(x), learn the RESIDUAL: F(x) = H(x) - x
So H(x) = x + F(x)

Why this works:

  1. GRADIENT HIGHWAY
     During backprop, gradient flows through the "+" unmodified
     Even if Sublayer gradient vanishes, the original gradient survives

     ∂L/∂x = ∂L/∂(x + F(x)) = ∂L/∂y + ∂L/∂y * ∂F/∂x
                               ↑          ↑
                         direct path   through sublayer

  2. IDENTITY INITIALIZATION
     If F(x) = 0 at init, output = x (identity function)
     Easy starting point - just pass through!
     Model can then learn to ADD refinements

  3. INFORMATION PRESERVATION
     Original input is always preserved
     Model learns: "Here's new context, but don't forget the original"
""")

    # Demonstrate gradient flow
    print("-" * 50)
    print("LAYER NORMALIZATION")
    print("-" * 50)

    print("""
The Problem: Activations can have different scales
  - Some neurons output values in [-100, 100]
  - Others in [-0.01, 0.01]
  - This makes optimization unstable

The Solution: Layer Normalization

  LayerNorm(x) = γ * (x - μ) / σ + β

  Where:
  - μ = mean across features (per token)
  - σ = std across features (per token)
  - γ, β = learned scale and shift (per feature)
""")

    # Show LayerNorm effect
    print("\nLayerNorm in action:")
    ln = nn.LayerNorm(d_model)

    # Create inputs with very different scales
    x_varied = mx.array([[1000.0] * 32 + [-1000.0] * 32])
    x_normed = ln(x_varied)

    print(f"  Input: mean={mx.mean(x_varied).item():.2f}, std={mx.std(x_varied).item():.2f}")
    print(f"  After LN: mean={mx.mean(x_normed).item():.4f}, std={mx.std(x_normed).item():.4f}")
    print("  (Normalized to ~mean 0, ~std 1)")

    print("\n" + "-" * 50)
    print("PRE-LN vs POST-LN")
    print("-" * 50)

    print("""
Original Paper (2017): Post-LN
  x → Attention → Add → LayerNorm → FFN → Add → LayerNorm

Modern Practice: Pre-LN
  x → LayerNorm → Attention → Add → LayerNorm → FFN → Add

Why Pre-LN is better:

  1. More stable gradients during training
  2. No need for learning rate warmup
  3. Can train deeper models more easily

The key insight: Normalizing BEFORE the sublayer keeps
activations bounded going into attention/FFN.
""")

    # Show full block diagram
    print("-" * 50)
    print("FULL TRANSFORMER BLOCK DIAGRAM")
    print("-" * 50)

    print("""
    ┌─────────────────────────────────────────────────────────┐
    │                   TRANSFORMER BLOCK                     │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │  Input x ─────────────────────────┐                     │
    │      │                            │                     │
    │      ▼                            │                     │
    │  ┌─────────┐                      │ (Residual)          │
    │  │LayerNorm│                      │                     │
    │  └────┬────┘                      │                     │
    │       │                           │                     │
    │       ▼                           │                     │
    │  ┌──────────────────┐             │                     │
    │  │ Multi-Head       │             │                     │
    │  │ Attention        │             │                     │
    │  └────────┬─────────┘             │                     │
    │           │                       │                     │
    │           ▼                       │                     │
    │         (+)◄──────────────────────┘                     │
    │           │                                             │
    │           ├──────────────────────┐                      │
    │           │                      │                      │
    │           ▼                      │                      │
    │  ┌─────────┐                     │ (Residual)           │
    │  │LayerNorm│                     │                      │
    │  └────┬────┘                     │                      │
    │       │                          │                      │
    │       ▼                          │                      │
    │  ┌──────────────────┐            │                      │
    │  │ Feed-Forward     │            │                      │
    │  │ Network          │            │                      │
    │  └────────┬─────────┘            │                      │
    │           │                      │                      │
    │           ▼                      │                      │
    │         (+)◄─────────────────────┘                      │
    │           │                                             │
    │           ▼                                             │
    │        Output                                           │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

    This block is repeated 6, 12, 24, 96+ times!
    (GPT-3 has 96 layers, each with this structure)
""")


if __name__ == "__main__":
    demo()
