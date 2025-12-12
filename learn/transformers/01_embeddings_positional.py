#!/usr/bin/env python3
"""
Step 1: Embeddings + Positional Encoding

The Transformer sees all words at once - no sequential processing.
This means "Man bites dog" and "Dog bites man" look IDENTICAL without position info.

Solution: Input = Word Embedding + Positional Encoding (summed, not concatenated)

Run: uv run python 01_embeddings_positional.py
"""

import mlx.core as mx
import mlx.nn as nn
import math


class TokenEmbedding(nn.Module):
    """Converts token IDs to dense vectors (captures semantic meaning)."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def __call__(self, x):
        # Scale by sqrt(d_model) - helps with gradient flow
        return self.embedding(x) * math.sqrt(self.d_model)


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> mx.array:
    """
    Creates positional encodings using sin/cos functions.

    Why sin/cos?
    - Deterministic (no learned parameters needed)
    - Can extrapolate to longer sequences than seen during training
    - Relative positions can be computed as linear functions

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    position = mx.arange(seq_len)[:, None]  # Shape: (seq_len, 1)
    dim = mx.arange(0, d_model, 2)  # Even indices: 0, 2, 4, ...

    # Compute the division term: 10000^(2i/d_model)
    div_term = mx.exp(dim * (-math.log(10000.0) / d_model))

    # Compute sin for even indices, cos for odd indices
    pe = mx.zeros((seq_len, d_model))

    # Even indices get sin
    sin_values = mx.sin(position * div_term)
    # Odd indices get cos
    cos_values = mx.cos(position * div_term)

    # Interleave sin and cos
    # For MLX, we'll construct this differently
    pe_even = sin_values
    pe_odd = cos_values

    # Stack and reshape to interleave
    pe = mx.concatenate([pe_even[:, :, None], pe_odd[:, :, None]], axis=2)
    pe = pe.reshape(seq_len, d_model)

    return pe


class LearnedPositionalEncoding(nn.Module):
    """
    Alternative: Learn position embeddings (like GPT).
    Simpler but can't extrapolate beyond max_seq_len.
    """

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

    def __call__(self, seq_len: int) -> mx.array:
        positions = mx.arange(seq_len)
        return self.pos_embedding(positions)


def demo():
    print("=" * 60)
    print("EMBEDDINGS + POSITIONAL ENCODING")
    print("=" * 60)

    # Hyperparameters
    vocab_size = 1000
    d_model = 64  # Embedding dimension (small for demo)
    seq_len = 5
    max_seq_len = 512

    # Simulate a sentence: "The bank of the river"
    # (Using fake token IDs)
    tokens = mx.array([10, 42, 15, 10, 88])  # [The, bank, of, the, river]
    print(f"\nInput tokens: {tokens.tolist()}")
    print(f"Sequence length: {seq_len}")
    print(f"Embedding dimension: {d_model}")

    # Step 1: Get word embeddings
    token_embed = TokenEmbedding(vocab_size, d_model)
    word_vectors = token_embed(tokens)
    print(f"\nWord embeddings shape: {word_vectors.shape}")
    print(f"Word embedding for 'bank' (token 42):\n  First 8 dims: {word_vectors[1, :8].tolist()}")

    # Step 2: Get positional encodings
    print("\n--- Sinusoidal Positional Encoding ---")
    pos_enc = sinusoidal_positional_encoding(seq_len, d_model)
    print(f"Positional encoding shape: {pos_enc.shape}")
    print(f"Position 0 encoding (first 8 dims): {pos_enc[0, :8].tolist()}")
    print(f"Position 4 encoding (first 8 dims): {pos_enc[4, :8].tolist()}")

    # Step 3: ADD them together (not concatenate!)
    input_vectors = word_vectors + pos_enc
    print(f"\n--- Combined Input ---")
    print(f"Final input shape: {input_vectors.shape}")
    print(f"Note: Same shape as word embeddings! (summed, not concatenated)")

    # Demonstrate why this works in high-dimensional space
    print("\n--- Why Addition Works ---")
    # The model learns to separate semantic and positional signals
    # In high-D space, random vectors are nearly orthogonal
    dot_products = []
    for i in range(seq_len):
        dot = mx.sum(word_vectors[i] * pos_enc[i]).item()
        norm_word = mx.sqrt(mx.sum(word_vectors[i] ** 2)).item()
        norm_pos = mx.sqrt(mx.sum(pos_enc[i] ** 2)).item()
        cosine_sim = dot / (norm_word * norm_pos + 1e-8)
        dot_products.append(cosine_sim)
    print(f"Cosine similarity between word & position vectors: {[f'{x:.3f}' for x in dot_products]}")
    print("(Low similarity = vectors are nearly orthogonal = easy to separate)")

    # Show learned positional encoding alternative
    print("\n--- Learned Positional Encoding (Alternative) ---")
    learned_pos = LearnedPositionalEncoding(max_seq_len, d_model)
    learned_enc = learned_pos(seq_len)
    print(f"Learned encoding shape: {learned_enc.shape}")
    print("(Used by GPT models - simpler but can't extrapolate)")


if __name__ == "__main__":
    demo()
