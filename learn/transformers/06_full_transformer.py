#!/usr/bin/env python3
"""
Step 6: Full Transformer (Encoder & Decoder)

Putting it all together:
- Encoder: Bidirectional, sees all words (BERT-style)
- Decoder: Causal/masked, can only see past words (GPT-style)

Run: uv run python 06_full_transformer.py
"""

import mlx.core as mx
import mlx.nn as nn
import math


# ============================================================
# Building Blocks (from previous files)
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
        batch_size, seq_len = x.shape[0], x.shape[1]
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch, heads, seq, d_k)

    def combine_heads(self, x):
        x = x.transpose(0, 2, 1, 3)  # (batch, seq, heads, d_k)
        batch_size, seq_len = x.shape[0], x.shape[1]
        return x.reshape(batch_size, seq_len, self.d_model)

    def __call__(self, q, k, v, mask=None):
        Q = self.split_heads(self.W_Q(q))
        K = self.split_heads(self.W_K(k))
        V = self.split_heads(self.W_V(v))
        attn_out, weights = scaled_dot_product_attention(Q, K, V, mask)
        return self.dropout(self.combine_heads(attn_out)), weights


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        return self.dropout(self.W2(nn.gelu(self.W1(x))))


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> mx.array:
    position = mx.arange(seq_len)[:, None]
    dim = mx.arange(0, d_model, 2)
    div_term = mx.exp(dim * (-math.log(10000.0) / d_model))
    pe_even = mx.sin(position * div_term)
    pe_odd = mx.cos(position * div_term)
    pe = mx.concatenate([pe_even[:, :, None], pe_odd[:, :, None]], axis=2)
    return pe.reshape(seq_len, d_model)


# ============================================================
# Encoder Block (Bidirectional)
# ============================================================


class EncoderBlock(nn.Module):
    """Single encoder block - can see ALL tokens (bidirectional)."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def __call__(self, x, mask=None):
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed, mask)
        x = x + attn_out

        # FFN with residual
        normed = self.norm2(x)
        x = x + self.ffn(normed)

        return x


class Encoder(nn.Module):
    """
    Transformer Encoder (like BERT).

    - Stacks N encoder blocks
    - Bidirectional: each token sees ALL other tokens
    - Great for: classification, NER, sentence embeddings
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = sinusoidal_positional_encoding(max_seq_len, d_model)

        self.layers = [
            EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ]
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x, mask=None):
        seq_len = x.shape[1]

        # Embeddings + positional encoding
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:seq_len]
        x = self.dropout(x)

        # Pass through all encoder layers
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


# ============================================================
# Decoder Block (Causal/Autoregressive)
# ============================================================


class DecoderBlock(nn.Module):
    """
    Single decoder block - can only see PAST tokens (causal).

    Has TWO attention mechanisms:
    1. Masked Self-Attention: Attend to previous output tokens only
    2. Cross-Attention: Attend to encoder output (if encoder-decoder model)
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()

        # Masked self-attention (causal)
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention (for encoder-decoder models)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def __call__(self, x, encoder_output=None, causal_mask=None, cross_mask=None):
        # Masked self-attention (can only see past)
        normed = self.norm1(x)
        attn_out, _ = self.self_attention(normed, normed, normed, causal_mask)
        x = x + attn_out

        # Cross-attention (if encoder output provided)
        if encoder_output is not None:
            normed = self.norm2(x)
            # Q from decoder, K and V from encoder
            cross_out, _ = self.cross_attention(normed, encoder_output, encoder_output, cross_mask)
            x = x + cross_out

        # FFN
        normed = self.norm3(x)
        x = x + self.ffn(normed)

        return x


class Decoder(nn.Module):
    """
    Transformer Decoder (like GPT).

    - Stacks N decoder blocks
    - Causal: each token can only see PREVIOUS tokens
    - Used for: text generation, language modeling
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = sinusoidal_positional_encoding(max_seq_len, d_model)

        self.layers = [
            DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ]
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Output projection to vocabulary
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

    def create_causal_mask(self, seq_len: int) -> mx.array:
        """Create causal mask: position i can only attend to positions <= i."""
        return mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))

    def __call__(self, x, encoder_output=None):
        batch_size, seq_len = x.shape

        # Embeddings + positional encoding
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:seq_len]
        x = self.dropout(x)

        # Create causal mask (can only see past)
        causal_mask = self.create_causal_mask(seq_len)

        # Pass through all decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, causal_mask)

        x = self.norm(x)

        # Project to vocabulary logits
        logits = self.output_proj(x)

        return logits


# ============================================================
# Full Encoder-Decoder Transformer (for translation, etc.)
# ============================================================


class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer (original "Attention is All You Need").

    Used for: Translation, Summarization, any seq2seq task

    Flow:
    1. Encoder processes source sequence (bidirectional)
    2. Decoder generates target sequence (causal)
    3. Decoder attends to encoder output via cross-attention
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size, d_model, num_heads, num_encoder_layers, d_ff, max_seq_len, dropout
        )
        self.decoder = Decoder(
            tgt_vocab_size, d_model, num_heads, num_decoder_layers, d_ff, max_seq_len, dropout
        )

    def __call__(self, src, tgt):
        encoder_output = self.encoder(src)
        logits = self.decoder(tgt, encoder_output)
        return logits


def demo():
    print("=" * 70)
    print("FULL TRANSFORMER: ENCODER & DECODER")
    print("=" * 70)

    # Hyperparameters (small for demo)
    vocab_size = 1000
    d_model = 64
    num_heads = 4
    num_layers = 2
    batch_size = 2
    seq_len = 8

    mx.random.seed(42)

    print("\n" + "=" * 70)
    print("PART 1: ENCODER (Bidirectional, like BERT)")
    print("=" * 70)

    print(f"""
Configuration:
  vocab_size: {vocab_size}
  d_model: {d_model}
  num_heads: {num_heads}
  num_layers: {num_layers}

The Encoder:
  - Sees ALL tokens simultaneously (bidirectional)
  - Used for understanding, classification, embeddings
  - Output: Contextual representations for each token
""")

    encoder = Encoder(vocab_size, d_model, num_heads, num_layers, dropout=0.0)

    # Random input tokens
    src_tokens = mx.random.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input tokens shape: {src_tokens.shape}")

    encoder_output = encoder(src_tokens)
    print(f"Encoder output shape: {encoder_output.shape}")
    print("  = (batch_size, seq_len, d_model)")
    print("  Each token now has rich contextual representation!")

    print("\n" + "=" * 70)
    print("PART 2: DECODER (Causal/Autoregressive, like GPT)")
    print("=" * 70)

    print(f"""
The Decoder:
  - Can only see PAST tokens (causal masking)
  - Used for generation, language modeling
  - Output: Probability distribution over next token
""")

    decoder = Decoder(vocab_size, d_model, num_heads, num_layers, dropout=0.0)

    # Random target tokens (what we're trying to generate)
    tgt_tokens = mx.random.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input tokens shape: {tgt_tokens.shape}")

    logits = decoder(tgt_tokens)
    print(f"Decoder output (logits) shape: {logits.shape}")
    print("  = (batch_size, seq_len, vocab_size)")
    print("  Each position predicts the NEXT token!")

    # Show causal mask
    print("\nCausal Mask (position i can see positions 0..i):")
    mask = decoder.create_causal_mask(5)
    print(mask.astype(mx.int32))

    print("\n" + "=" * 70)
    print("PART 3: ENCODER-DECODER (Translation, like original Transformer)")
    print("=" * 70)

    print("""
For sequence-to-sequence tasks (translation, summarization):
  1. Encoder reads source: "The cat sat on the mat"
  2. Decoder generates target: "Le chat était assis sur le tapis"
  3. Decoder CROSS-ATTENDS to encoder output

The decoder can "look at" the full source sentence while generating!
""")

    transformer = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        dropout=0.0,
    )

    src = mx.random.randint(0, vocab_size, (batch_size, seq_len))
    tgt = mx.random.randint(0, vocab_size, (batch_size, seq_len))

    logits = transformer(src, tgt)
    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Output logits shape: {logits.shape}")

    print("\n" + "=" * 70)
    print("SUMMARY: THE COMPLETE FLOW")
    print("=" * 70)

    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  INPUT TOKENS                         TARGET TOKENS                 │
│      │                                     │                        │
│      ▼                                     ▼                        │
│  ┌─────────────┐                    ┌─────────────┐                │
│  │ Token Embed │                    │ Token Embed │                │
│  └──────┬──────┘                    └──────┬──────┘                │
│         │                                  │                        │
│         + Positional Encoding              + Positional Encoding    │
│         │                                  │                        │
│         ▼                                  ▼                        │
│  ┌─────────────────┐              ┌─────────────────────┐          │
│  │    ENCODER      │              │      DECODER        │          │
│  │  (Bidirectional)│              │     (Causal)        │          │
│  ├─────────────────┤              ├─────────────────────┤          │
│  │                 │              │                     │          │
│  │  Self-Attention │   ──────►    │  Masked Self-Attn   │          │
│  │       +         │   encoder    │         +           │          │
│  │     FFN         │   output     │  Cross-Attention ◄──┘          │
│  │       ×N        │              │         +           │          │
│  │                 │              │       FFN           │          │
│  └────────┬────────┘              │         ×N          │          │
│           │                       │                     │          │
│           │                       └──────────┬──────────┘          │
│           │                                  │                      │
│           │                                  ▼                      │
│           │                       ┌─────────────────────┐          │
│           │                       │   Linear + Softmax  │          │
│           │                       │   (vocab_size out)  │          │
│           │                       └──────────┬──────────┘          │
│           │                                  │                      │
│           ▼                                  ▼                      │
│  Contextual Embeddings            Next Token Probabilities         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

KEY DIFFERENCES:

  ENCODER (BERT-style)           DECODER (GPT-style)
  ─────────────────────          ──────────────────────
  • Bidirectional                • Causal (left-to-right)
  • Sees all tokens              • Only sees past tokens
  • Understanding tasks          • Generation tasks
  • Classification, NER          • Text completion, chat
  • No autoregressive gen        • Autoregressive generation


MODERN TRENDS:

  • GPT family: Decoder-only (no encoder)
  • BERT family: Encoder-only (no decoder)
  • T5, BART: Full encoder-decoder
  • Most LLMs today: Decoder-only with massive scale

The magic: All this is learned through backpropagation!
  W_Q, W_K, W_V, FFN weights → learned to minimize prediction error
""")

    # Parameter count
    print("\n" + "=" * 70)
    print("PARAMETER COUNT")
    print("=" * 70)

    def count_params(model):
        return sum(p.size for p in model.parameters().values())

    enc_params = count_params(encoder)
    dec_params = count_params(decoder)
    full_params = count_params(transformer)

    print(f"""
This demo model:
  Encoder: {enc_params:,} parameters
  Decoder: {dec_params:,} parameters
  Full Transformer: {full_params:,} parameters

Real models:
  GPT-2:    117M - 1.5B parameters
  GPT-3:    175B parameters
  GPT-4:    ~1.8T parameters (estimated)
  LLaMA-2:  7B - 70B parameters
  Claude:   Undisclosed (very large)

Same architecture, just scaled up!
  • More layers (12 → 96+)
  • Bigger d_model (768 → 12288+)
  • More heads (12 → 96+)
  • Larger vocab (50k → 100k+)
""")


if __name__ == "__main__":
    demo()
