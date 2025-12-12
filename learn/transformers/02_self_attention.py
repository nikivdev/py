#!/usr/bin/env python3
"""
Step 2: Self-Attention (Query, Key, Value)

The heart of the Transformer. This is what allows "Bank" to understand
it means "riverbank" not "financial bank" based on context.

The File Cabinet Analogy:
- Query (Q): The sticky note with what you're looking for
- Key (K):   The label on folder tabs (for matching)
- Value (V): The actual papers inside the folder

Run: uv run python 02_self_attention.py
"""

import mlx.core as mx
import mlx.nn as nn
import math


def scaled_dot_product_attention(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    mask: mx.array | None = None,
    verbose: bool = False,
) -> tuple[mx.array, mx.array]:
    """
    The core attention formula:

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Step by step:
    1. QK^T: How similar is each query to each key? (dot product)
    2. / sqrt(d_k): Scale down to prevent softmax saturation
    3. softmax: Convert to probabilities (attention weights)
    4. * V: Weighted sum of values
    """
    d_k = K.shape[-1]

    # Step 1: Compute attention scores (QK^T)
    # Q: (seq_len, d_k), K: (seq_len, d_k)
    # scores: (seq_len, seq_len) - how much each word attends to each other word
    scores = mx.matmul(Q, K.T)

    if verbose:
        print(f"\n  Raw attention scores (QK^T):")
        print(f"  Shape: {scores.shape}")
        print(f"  {scores}")

    # Step 2: Scale by sqrt(d_k)
    # Why? Without scaling, dot products grow with dimension,
    # pushing softmax into regions with tiny gradients
    scores = scores / math.sqrt(d_k)

    if verbose:
        print(f"\n  Scaled scores (/ sqrt({d_k})):")
        print(f"  {scores}")

    # Step 3: Apply mask (if provided) - for decoder/causal attention
    if mask is not None:
        # Set masked positions to -inf so softmax gives 0
        scores = mx.where(mask, scores, mx.array(float("-inf")))
        if verbose:
            print(f"\n  Masked scores:")
            print(f"  {scores}")

    # Step 4: Softmax to get attention weights (probabilities)
    attention_weights = mx.softmax(scores, axis=-1)

    if verbose:
        print(f"\n  Attention weights (softmax):")
        print(f"  {attention_weights}")
        print(f"  Row sums: {mx.sum(attention_weights, axis=-1)}")  # Should be 1.0

    # Step 5: Weighted sum of values
    output = mx.matmul(attention_weights, V)

    return output, attention_weights


class SelfAttention(nn.Module):
    """
    Self-attention: Q, K, V all come from the same sequence.

    For each word vector x, we project it THREE times:
    - Q = x @ W_Q  (What am I looking for?)
    - K = x @ W_K  (What do I contain? For matching)
    - V = x @ W_V  (What is my actual information?)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # These are the LEARNED weight matrices
        # The model learns HOW to pay attention during training
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

    def __call__(
        self, x: mx.array, mask: mx.array | None = None, verbose: bool = False
    ) -> tuple[mx.array, mx.array]:
        # Project input to Q, K, V
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        if verbose:
            print(f"\nProjections:")
            print(f"  Q shape: {Q.shape}")
            print(f"  K shape: {K.shape}")
            print(f"  V shape: {V.shape}")

        # Compute attention
        output, weights = scaled_dot_product_attention(Q, K, V, mask, verbose)

        return output, weights


def demo():
    print("=" * 60)
    print("SELF-ATTENTION: QUERY, KEY, VALUE")
    print("=" * 60)

    # Small dimensions for interpretability
    d_model = 8
    seq_len = 4

    # Simulate: "The bank of river"
    # Create fake input embeddings (normally from Step 1)
    mx.random.seed(42)
    x = mx.random.normal((seq_len, d_model))

    word_labels = ["The", "bank", "of", "river"]
    print(f"\nInput sequence: {word_labels}")
    print(f"Input shape: {x.shape} (seq_len={seq_len}, d_model={d_model})")

    # Create self-attention module
    attention = SelfAttention(d_model)

    print("\n" + "-" * 50)
    print("STEP-BY-STEP ATTENTION COMPUTATION")
    print("-" * 50)

    output, weights = attention(x, verbose=True)

    print("\n" + "-" * 50)
    print("INTERPRETING ATTENTION WEIGHTS")
    print("-" * 50)

    print("\nAttention weight matrix interpretation:")
    print("  Row i = what word i attends TO")
    print("  Column j = how much word i attends to word j")
    print()

    for i, word in enumerate(word_labels):
        print(f"  '{word}' attends to:")
        for j, target in enumerate(word_labels):
            weight = weights[i, j].item()
            bar = "█" * int(weight * 20)
            print(f"    {target:6s}: {weight:.3f} {bar}")
        print()

    # Demonstrate the "Bank" example from the explanation
    print("-" * 50)
    print("THE 'BANK' DISAMBIGUATION EXAMPLE")
    print("-" * 50)

    # In a trained model, "bank" would attend highly to "river"
    # Here we simulate what SHOULD happen with good weights
    print("\nIn a trained model processing 'The bank of the river':")
    print("  Q_bank · K_river → HIGH (semantically related)")
    print("  Q_bank · K_the   → LOW  (not informative)")
    print()
    print("After softmax, 'bank' might have attention weights like:")
    print("  river: 0.85  ← Strongly attends (disambiguates meaning!)")
    print("  of:    0.05")
    print("  the:   0.05")
    print("  bank:  0.05  (self-attention)")
    print()
    print("The new vector for 'bank' becomes:")
    print("  0.85 * V_river + 0.05 * V_of + 0.05 * V_the + 0.05 * V_bank")
    print("  = 'Bank' now encodes 'riverbank' meaning, not 'financial bank'!")

    # Causal/Decoder masking demo
    print("\n" + "-" * 50)
    print("CAUSAL MASKING (For Decoders like GPT)")
    print("-" * 50)

    # Create causal mask: word i can only see words 0..i
    # True = can attend, False = cannot attend
    causal_mask = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))
    print(f"\nCausal mask (True = can see, False = masked):")
    print(causal_mask)
    print()
    print("Word 0 can see: [0]")
    print("Word 1 can see: [0, 1]")
    print("Word 2 can see: [0, 1, 2]")
    print("Word 3 can see: [0, 1, 2, 3]")

    output_causal, weights_causal = attention(x, mask=causal_mask, verbose=False)
    print(f"\nCausal attention weights:")
    print(weights_causal)
    print("(Notice: upper triangle is 0 - future words are hidden!)")


if __name__ == "__main__":
    demo()
