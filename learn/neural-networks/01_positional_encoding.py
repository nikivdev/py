#!/usr/bin/env python3
"""
Positional Encoding - The Transformer's Sense of Order

WHY DO WE NEED THIS?
====================
Transformers process all tokens in parallel - they see "The cat sat" all at once.
Without positional info, "cat sat the" looks identical to "the sat cat".
Positional encoding adds a unique "signature" to each position.

THE INTUITION
=============
Think of a clock: second hand (fast), minute hand (medium), hour hand (slow).
By combining all hands, you uniquely identify any time.
Similarly, we use waves at different frequencies - fast-changing for early dimensions,
slow-changing for later dimensions.

THE MATH
========
For position `pos` and dimension `i`:
    PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))  # even dims
    PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))  # odd dims

Run: uv run python -m learn.neural-networks.01_positional_encoding
"""

import mlx.core as mx


def positional_encoding(seq_len: int, d_model: int) -> mx.array:
    """
    Generate sinusoidal positional encodings.

    Args:
        seq_len: Number of positions (e.g., max sequence length)
        d_model: Embedding dimension

    Returns:
        Array of shape (seq_len, d_model) with positional encodings
    """
    # Position indices: [0, 1, 2, ..., seq_len-1]
    pos = mx.arange(seq_len)[:, None]  # Shape: (seq_len, 1)

    # Dimension indices for the formula
    # For dims [0,1,2,3,4,5], we need i = [0,0,1,1,2,2]
    dim = mx.arange(d_model)[None, :]  # Shape: (1, d_model)

    # Compute the divisor: 10000^(2i/d_model)
    # (dim // 2) * 2 gives us [0,0,2,2,4,4,...] which is "2i" for each dimension
    div_term = 10000 ** ((dim // 2) * 2 / d_model)

    # Compute angles: pos / divisor
    # Broadcasting: (seq_len, 1) / (1, d_model) -> (seq_len, d_model)
    angles = pos / div_term

    # Apply sin to even indices, cos to odd indices
    # Create the encoding array
    pe = mx.zeros((seq_len, d_model))

    # Even dimensions (0, 2, 4, ...) get sin
    pe = pe.at[:, 0::2].add(mx.sin(angles[:, 0::2]))

    # Odd dimensions (1, 3, 5, ...) get cos
    pe = pe.at[:, 1::2].add(mx.cos(angles[:, 1::2]))

    return pe


def positional_encoding_v2(seq_len: int, d_model: int) -> mx.array:
    """
    Alternative implementation - more explicit about the math.
    """
    pe = mx.zeros((seq_len, d_model))

    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            # The wavelength increases with dimension
            div_term = 10000 ** (i / d_model)

            # Even dimension: sin
            pe = pe.at[pos, i].add(mx.sin(pos / div_term))

            # Odd dimension: cos (if it exists)
            if i + 1 < d_model:
                pe = pe.at[pos, i + 1].add(mx.cos(pos / div_term))

    return pe


def visualize_encoding(pe: mx.array):
    """Print a visual representation of the positional encoding."""
    seq_len, d_model = pe.shape
    print(f"\nPositional Encoding: seq_len={seq_len}, d_model={d_model}")
    print("=" * 70)

    # Header
    header = "pos |"
    for d in range(min(d_model, 8)):
        freq = "fast" if d < 2 else ("med" if d < 4 else "slow")
        kind = "sin" if d % 2 == 0 else "cos"
        header += f" d{d}({kind},{freq}) |"
    if d_model > 8:
        header += " ..."
    print(header)
    print("-" * 70)

    # Values
    for pos in range(seq_len):
        row = f" {pos}  |"
        for d in range(min(d_model, 8)):
            val = float(pe[pos, d])
            row += f"   {val:+.3f}   |"
        if d_model > 8:
            row += " ..."
        print(row)


def show_uniqueness(pe: mx.array):
    """Demonstrate that each position has a unique encoding."""
    print("\n\nUNIQUENESS CHECK")
    print("=" * 50)
    print("Each position should have a unique 'fingerprint'")
    print("Let's compute pairwise distances:\n")

    seq_len = pe.shape[0]
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            # Euclidean distance between position i and j
            diff = pe[i] - pe[j]
            dist = float(mx.sqrt(mx.sum(diff * diff)))
            print(f"  Distance(pos {i}, pos {j}) = {dist:.4f}")


def show_relative_positions(pe: mx.array):
    """Show that relative positions have consistent relationships."""
    print("\n\nRELATIVE POSITION PROPERTY")
    print("=" * 50)
    print("Key insight: PE[pos+k] can be computed as a linear function of PE[pos]")
    print("This means the model can easily learn to attend to relative positions!\n")

    # For any fixed offset k, PE[pos+k] = T_k @ PE[pos] for some rotation matrix T_k
    # Let's verify by checking dot products
    seq_len = pe.shape[0]

    print("Dot products between adjacent positions:")
    for pos in range(seq_len - 1):
        dot = float(mx.sum(pe[pos] * pe[pos + 1]))
        print(f"  PE[{pos}] . PE[{pos+1}] = {dot:.4f}")

    print("\nNotice: adjacent positions have similar (but not identical) dot products!")
    print("This consistency helps the model learn relative position relationships.")


def main():
    print("POSITIONAL ENCODING - Learning with MLX")
    print("=" * 70)

    # Small example for visualization
    seq_len = 6
    d_model = 8

    print(f"\nGenerating positional encoding for {seq_len} positions, {d_model} dimensions...")

    pe = positional_encoding(seq_len, d_model)

    # Force computation and show values
    mx.eval(pe)

    visualize_encoding(pe)
    show_uniqueness(pe)
    show_relative_positions(pe)

    # Verify both implementations match
    print("\n\nVERIFICATION")
    print("=" * 50)
    pe_v2 = positional_encoding_v2(seq_len, d_model)
    mx.eval(pe_v2)
    diff = float(mx.max(mx.abs(pe - pe_v2)))
    print(f"Max difference between vectorized and loop implementation: {diff:.10f}")
    print("(Should be ~0, proving they're equivalent)")

    # Larger example to show MLX speed
    print("\n\nPERFORMANCE TEST")
    print("=" * 50)
    import time

    large_seq = 2048
    large_dim = 512

    start = time.perf_counter()
    large_pe = positional_encoding(large_seq, large_dim)
    mx.eval(large_pe)  # Force computation
    elapsed = time.perf_counter() - start

    print(f"Generated ({large_seq}, {large_dim}) encoding in {elapsed*1000:.2f}ms")
    print(f"Shape: {large_pe.shape}")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("1. Each position gets a unique encoding (like a fingerprint)")
    print("2. Early dimensions change fast (high freq), later ones change slow")
    print("3. Relative positions have consistent relationships (helps attention)")
    print("4. No learned parameters - deterministic and generalizes to any length!")
    print("=" * 70)


if __name__ == "__main__":
    main()
