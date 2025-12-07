#!/usr/bin/env python3
"""
Self-Attention - The Heart of the Transformer

WHAT IS ATTENTION?
==================
Attention lets each token "look at" all other tokens and decide which ones are relevant.
For "The cat sat on the mat", when processing "sat", attention might focus heavily
on "cat" (who sat?) and "mat" (where?).

THE INTUITION
=============
Think of attention as a soft database lookup:
- Query (Q): "What am I looking for?"
- Key (K): "What do I contain?"
- Value (V): "What information do I provide?"

Each token asks "who's relevant to me?" by comparing its Query to all Keys,
then takes a weighted sum of Values.

THE MATH
========
1. Project input X into Q, K, V using learned weights
2. Compute attention scores: scores = Q @ K.T / sqrt(d_k)
3. Apply softmax to get attention weights (sum to 1)
4. Output = attention_weights @ V

Run: uv run python -m learn.neural-networks.02_attention
Or:  flow run nn 02_attention
"""

import mlx.core as mx


def softmax(x: mx.array, axis: int = -1) -> mx.array:
    """Numerically stable softmax."""
    x_max = mx.max(x, axis=axis, keepdims=True)
    exp_x = mx.exp(x - x_max)
    return exp_x / mx.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    mask: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """
    Core attention mechanism.

    Args:
        Q: Queries, shape (batch, seq_len, d_k) or (seq_len, d_k)
        K: Keys, shape (batch, seq_len, d_k) or (seq_len, d_k)
        V: Values, shape (batch, seq_len, d_v) or (seq_len, d_v)
        mask: Optional mask to prevent attending to certain positions

    Returns:
        output: Weighted sum of values, shape (batch, seq_len, d_v)
        attention_weights: The attention distribution, shape (batch, seq_len, seq_len)
    """
    d_k = Q.shape[-1]

    # Step 1: Compute attention scores
    # Q @ K.T gives us (seq_len, seq_len) - how much each position attends to each other
    scores = Q @ mx.swapaxes(K, -2, -1)  # (..., seq_len, seq_len)

    # Step 2: Scale by sqrt(d_k)
    # Without scaling, dot products grow with d_k, making softmax too peaky
    scores = scores / mx.sqrt(mx.array(d_k, dtype=scores.dtype))

    # Step 3: Apply mask (optional) - for causal attention or padding
    if mask is not None:
        # Set masked positions to -inf so softmax gives them 0 weight
        scores = mx.where(mask, scores, mx.array(float("-inf")))

    # Step 4: Softmax to get attention weights (each row sums to 1)
    attention_weights = softmax(scores, axis=-1)

    # Step 5: Weighted sum of values
    output = attention_weights @ V

    return output, attention_weights


def create_causal_mask(seq_len: int) -> mx.array:
    """
    Create a causal (autoregressive) mask.
    Position i can only attend to positions <= i.
    Used in decoder / language models.
    """
    # Lower triangular matrix of True values
    mask = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))
    return mask


def visualize_attention(weights: mx.array, tokens: list[str]):
    """Pretty print attention weights as a matrix."""
    seq_len = len(tokens)
    print("\nAttention Weights (rows=query, cols=key):")
    print("-" * (12 + seq_len * 8))

    # Header
    header = "         |"
    for tok in tokens:
        header += f" {tok[:6]:^6}|"
    print(header)
    print("-" * (12 + seq_len * 8))

    # Rows
    for i, tok in enumerate(tokens):
        row = f" {tok[:6]:>6}  |"
        for j in range(seq_len):
            w = float(weights[i, j])
            # Use intensity markers
            if w > 0.5:
                marker = "###"
            elif w > 0.2:
                marker = "## "
            elif w > 0.1:
                marker = "#  "
            else:
                marker = "   "
            row += f" {w:.2f}{marker}|"
        print(row)
    print("-" * (12 + seq_len * 8))


def demo_simple_attention():
    """Basic self-attention without learned projections."""
    print("\n" + "=" * 70)
    print("DEMO 1: Simple Self-Attention")
    print("=" * 70)

    # Simulate 4 tokens with 3-dimensional embeddings
    tokens = ["The", "cat", "sat", "mat"]
    seq_len = 4
    d_model = 8

    # Random embeddings (in practice, these come from an embedding layer)
    mx.random.seed(42)
    X = mx.random.normal((seq_len, d_model))

    print(f"\nInput: {tokens}")
    print(f"Embedding shape: {X.shape} (seq_len={seq_len}, d_model={d_model})")

    # In simplest case, Q=K=V=X (no projection)
    output, weights = scaled_dot_product_attention(X, X, X)
    mx.eval(output, weights)

    visualize_attention(weights, tokens)

    print("\nInterpretation:")
    print("- Each row shows how much that token attends to others")
    print("- Row sum = 1.0 (it's a probability distribution)")
    print("- High values mean 'this token is relevant to me'")


def demo_causal_attention():
    """Causal (autoregressive) attention for language modeling."""
    print("\n" + "=" * 70)
    print("DEMO 2: Causal Self-Attention (for Language Models)")
    print("=" * 70)

    tokens = ["The", "cat", "sat", "mat"]
    seq_len = 4
    d_model = 8

    mx.random.seed(42)
    X = mx.random.normal((seq_len, d_model))

    # Create causal mask - each position can only see previous positions
    mask = create_causal_mask(seq_len)

    print(f"\nCausal Mask (True = can attend, False = blocked):")
    for i, tok in enumerate(tokens):
        row = f"  {tok}: ["
        for j in range(seq_len):
            row += "T " if bool(mask[i, j]) else "- "
        row += "]"
        print(row)

    output, weights = scaled_dot_product_attention(X, X, X, mask=mask)
    mx.eval(output, weights)

    visualize_attention(weights, tokens)

    print("\nInterpretation:")
    print("- 'The' can only see itself (first token)")
    print("- 'mat' can see all previous tokens")
    print("- This prevents 'cheating' during autoregressive generation!")


def demo_qkv_projections():
    """Attention with learned Q, K, V projections."""
    print("\n" + "=" * 70)
    print("DEMO 3: Attention with Q, K, V Projections")
    print("=" * 70)

    tokens = ["The", "cat", "sat", "mat"]
    seq_len = 4
    d_model = 8
    d_k = 4  # Dimension of Q, K (can differ from d_model)
    d_v = 4  # Dimension of V

    mx.random.seed(42)
    X = mx.random.normal((seq_len, d_model))

    # Learned projection matrices (in practice, these are trained)
    W_Q = mx.random.normal((d_model, d_k)) * 0.1
    W_K = mx.random.normal((d_model, d_k)) * 0.1
    W_V = mx.random.normal((d_model, d_v)) * 0.1

    # Project inputs to Q, K, V spaces
    Q = X @ W_Q  # (seq_len, d_k)
    K = X @ W_K  # (seq_len, d_k)
    V = X @ W_V  # (seq_len, d_v)

    print(f"\nInput X:     shape {X.shape}")
    print(f"Projected Q: shape {Q.shape}")
    print(f"Projected K: shape {K.shape}")
    print(f"Projected V: shape {V.shape}")

    output, weights = scaled_dot_product_attention(Q, K, V)
    mx.eval(output, weights)

    print(f"Output:      shape {output.shape}")

    visualize_attention(weights, tokens)

    print("\nWhy project?")
    print("- Q projection: 'What am I looking for?'")
    print("- K projection: 'What do I offer to queries?'")
    print("- V projection: 'What information do I pass along?'")
    print("- These projections are LEARNED - the model discovers what to attend to!")


def demo_multi_head_attention():
    """Multi-head attention - multiple attention patterns in parallel."""
    print("\n" + "=" * 70)
    print("DEMO 4: Multi-Head Attention")
    print("=" * 70)

    tokens = ["The", "cat", "sat", "mat"]
    seq_len = 4
    d_model = 8
    n_heads = 2
    d_k = d_model // n_heads  # 4

    mx.random.seed(42)
    X = mx.random.normal((seq_len, d_model))

    print(f"\nInput: {X.shape}")
    print(f"Number of heads: {n_heads}")
    print(f"Dimension per head: {d_k}")

    # Each head gets its own Q, K, V projections
    all_outputs = []
    for h in range(n_heads):
        print(f"\n--- Head {h + 1} ---")

        # Project to smaller dimension for this head
        W_Q = mx.random.normal((d_model, d_k)) * 0.1
        W_K = mx.random.normal((d_model, d_k)) * 0.1
        W_V = mx.random.normal((d_model, d_k)) * 0.1

        Q = X @ W_Q
        K = X @ W_K
        V = X @ W_V

        head_out, weights = scaled_dot_product_attention(Q, K, V)
        mx.eval(head_out, weights)
        all_outputs.append(head_out)

        visualize_attention(weights, tokens)

    # Concatenate heads and project back
    concat = mx.concatenate(all_outputs, axis=-1)  # (seq_len, d_model)
    W_O = mx.random.normal((d_model, d_model)) * 0.1
    output = concat @ W_O

    mx.eval(output)
    print(f"\nConcatenated heads: {concat.shape}")
    print(f"Final output: {output.shape}")

    print("\nWhy multi-head?")
    print("- Different heads can learn different attention patterns")
    print("- Head 1 might focus on syntax (subject-verb)")
    print("- Head 2 might focus on semantics (related concepts)")
    print("- More expressive than single attention!")


def main():
    print("SELF-ATTENTION - Learning with MLX")
    print("=" * 70)
    print("\nAttention is the mechanism that lets each token 'look at' all others")
    print("and decide which information is relevant.")

    demo_simple_attention()
    demo_causal_attention()
    demo_qkv_projections()
    demo_multi_head_attention()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("1. Attention computes relevance scores between all token pairs")
    print("2. Softmax ensures weights sum to 1 (probability distribution)")
    print("3. Scaling by sqrt(d_k) prevents vanishing gradients")
    print("4. Causal mask enables autoregressive (left-to-right) generation")
    print("5. Q/K/V projections let the model LEARN what to attend to")
    print("6. Multi-head attention captures multiple relationship types")
    print("=" * 70)


if __name__ == "__main__":
    main()
