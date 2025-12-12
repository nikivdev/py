#!/usr/bin/env python3
"""
Step 3: Multi-Head Attention

Why multiple heads? A word has MULTIPLE relationships:
- "Bank" → "River" (semantic meaning)
- "Bank" → is a noun (syntactic role)
- "Bank" → is the subject (grammatical function)

Each head learns to focus on DIFFERENT types of relationships.
We run attention 8, 12, or 96 times in parallel!

Run: uv run python 03_multi_head_attention.py
"""

import mlx.core as mx
import mlx.nn as nn
import math


def scaled_dot_product_attention(
    Q: mx.array, K: mx.array, V: mx.array, mask: mx.array | None = None
) -> tuple[mx.array, mx.array]:
    """Same as before, but handles batched heads."""
    d_k = K.shape[-1]
    scores = mx.matmul(Q, K.swapaxes(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = mx.where(mask, scores, mx.array(float("-inf")))

    weights = mx.softmax(scores, axis=-1)
    output = mx.matmul(weights, V)

    return output, weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention runs self-attention h times in parallel.

    The key insight:
    - Split d_model into h heads, each with d_k = d_model / h dimensions
    - Each head learns different attention patterns
    - Concatenate results and project back

    MultiHead(Q,K,V) = Concat(head_1, ..., head_h) @ W_O
    where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Projections for Q, K, V (could be separate per head, but this is equivalent)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        # Output projection: combines all heads back to d_model
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x: mx.array) -> mx.array:
        """
        Split the last dimension into (num_heads, d_k).
        Input:  (seq_len, d_model)
        Output: (num_heads, seq_len, d_k)
        """
        seq_len = x.shape[0]
        # Reshape: (seq_len, d_model) -> (seq_len, num_heads, d_k)
        x = x.reshape(seq_len, self.num_heads, self.d_k)
        # Transpose: (seq_len, num_heads, d_k) -> (num_heads, seq_len, d_k)
        return x.transpose(1, 0, 2)

    def combine_heads(self, x: mx.array) -> mx.array:
        """
        Reverse of split_heads.
        Input:  (num_heads, seq_len, d_k)
        Output: (seq_len, d_model)
        """
        # Transpose: (num_heads, seq_len, d_k) -> (seq_len, num_heads, d_k)
        x = x.transpose(1, 0, 2)
        seq_len = x.shape[0]
        # Reshape: (seq_len, num_heads, d_k) -> (seq_len, d_model)
        return x.reshape(seq_len, self.d_model)

    def __call__(
        self, x: mx.array, mask: mx.array | None = None, verbose: bool = False
    ) -> tuple[mx.array, mx.array]:
        # Step 1: Project to Q, K, V
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        if verbose:
            print(f"\n  After projection: Q, K, V all have shape {Q.shape}")

        # Step 2: Split into multiple heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        if verbose:
            print(f"  After splitting into {self.num_heads} heads: {Q.shape}")
            print(f"  Each head processes d_k = {self.d_k} dimensions")

        # Step 3: Apply attention to each head in parallel
        # The batch dimension handles all heads at once!
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        if verbose:
            print(f"  Attention output per head: {attn_output.shape}")

        # Step 4: Concatenate heads
        combined = self.combine_heads(attn_output)

        if verbose:
            print(f"  After concatenating heads: {combined.shape}")

        # Step 5: Final linear projection
        output = self.W_O(combined)

        if verbose:
            print(f"  After output projection: {output.shape}")

        return output, attn_weights


def demo():
    print("=" * 60)
    print("MULTI-HEAD ATTENTION")
    print("=" * 60)

    d_model = 64
    num_heads = 8
    seq_len = 5

    mx.random.seed(42)
    x = mx.random.normal((seq_len, d_model))

    word_labels = ["The", "bank", "of", "the", "river"]
    print(f"\nInput sequence: {word_labels}")
    print(f"d_model: {d_model}")
    print(f"num_heads: {num_heads}")
    print(f"d_k (dim per head): {d_model // num_heads}")

    # Create multi-head attention
    mha = MultiHeadAttention(d_model, num_heads)

    print("\n" + "-" * 50)
    print("MULTI-HEAD ATTENTION COMPUTATION")
    print("-" * 50)

    output, attn_weights = mha(x, verbose=True)

    print("\n" + "-" * 50)
    print("WHAT EACH HEAD LEARNS")
    print("-" * 50)

    print(f"\nAttention weights shape: {attn_weights.shape}")
    print(f"  = ({num_heads} heads, {seq_len} seq_len, {seq_len} seq_len)")

    print("\nVisualization of what different heads attend to:")
    print("(In a trained model, each head specializes)")
    print()

    # Show attention patterns for first 3 heads
    for head_idx in range(min(3, num_heads)):
        print(f"--- Head {head_idx + 1} ---")
        head_weights = attn_weights[head_idx]

        # Find what "bank" (index 1) attends to in this head
        bank_attention = head_weights[1]
        max_idx = int(mx.argmax(bank_attention).item())

        print(f"  'bank' most attends to: '{word_labels[max_idx]}'")
        print(f"  Full attention from 'bank': ", end="")
        for j, word in enumerate(word_labels):
            print(f"{word}:{bank_attention[j].item():.2f} ", end="")
        print("\n")

    print("-" * 50)
    print("WHY MULTIPLE HEADS MATTER")
    print("-" * 50)

    print("""
In a trained transformer:

  Head 1 might learn: SEMANTIC relationships
    "bank" → "river" (meaning)
    "bank" → "money" (in financial context)

  Head 2 might learn: SYNTACTIC relationships
    "bank" → "the" (determiner of noun)
    "runs" → "he" (subject of verb)

  Head 3 might learn: POSITIONAL patterns
    word[i] → word[i-1] (previous word)
    word[i] → word[i+1] (next word)

  Head 4 might learn: COREFERENCE
    "it" → "the dog" (pronoun resolution)
    "they" → "the students"

The model learns WHICH relationships are useful for the task!
Each head is like a specialist examining the sentence differently.

After attention:
  - Concatenate: [head1_output | head2_output | ... | head_h_output]
  - Project with W_O to mix information from all heads
  - Result: rich representation encoding multiple relationship types
""")

    # Show dimension math
    print("-" * 50)
    print("DIMENSION MATH")
    print("-" * 50)
    print(f"""
  Input:  ({seq_len}, {d_model})
  Split:  ({num_heads}, {seq_len}, {d_model // num_heads})  # Each head gets d_k dims
  Attn:   ({num_heads}, {seq_len}, {d_model // num_heads})  # Same shape after attention
  Concat: ({seq_len}, {d_model})                    # Back to original
  W_O:    ({seq_len}, {d_model})                    # Final output

  Total parameters in Multi-Head Attention:
  - W_Q: {d_model} × {d_model} = {d_model * d_model}
  - W_K: {d_model} × {d_model} = {d_model * d_model}
  - W_V: {d_model} × {d_model} = {d_model * d_model}
  - W_O: {d_model} × {d_model} = {d_model * d_model}
  - Total: {4 * d_model * d_model} parameters
""")


if __name__ == "__main__":
    demo()
