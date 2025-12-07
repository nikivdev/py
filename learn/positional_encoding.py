#!/usr/bin/env python3
"""
Positional Encoding - Interactive Exploration

This script lets you build up positional encoding step by step,
seeing exactly what each operation does.

Run: uv run python learn/positional_encoding.py
"""

import numpy as np


def step_by_step_positional_encoding(seq_len: int = 4, d_model: int = 4):
    """Build positional encoding step by step with explanations."""

    print("=" * 70)
    print(f"POSITIONAL ENCODING: seq_len={seq_len}, d_model={d_model}")
    print("=" * 70)

    # =========================================================================
    # STEP 1: Create the position vector
    # =========================================================================
    print("\n" + "─" * 70)
    print("STEP 1: Create position vector")
    print("─" * 70)

    position = np.arange(seq_len, dtype=np.float32)[:, np.newaxis]

    print(f"\nposition = np.arange({seq_len})[:, np.newaxis]")
    print(f"\nShape: {position.shape}")
    print(f"Values:\n{position}")
    print("\n→ Each row is a position in the sequence (0, 1, 2, ...)")
    print("→ The [:, np.newaxis] adds a dimension for broadcasting later")

    # =========================================================================
    # STEP 2: Create the division term (denominator)
    # =========================================================================
    print("\n" + "─" * 70)
    print("STEP 2: Create division term (the wavelength controller)")
    print("─" * 70)

    # Generate even indices: [0, 2, 4, ...]
    div_indices = np.arange(0, d_model, 2, dtype=np.float32)

    print(f"\ndiv_indices = np.arange(0, {d_model}, 2)")
    print(f"Values: {div_indices}")
    print("\n→ These are the 'i' values in the formula: 2i = 0, 2, 4, ...")
    print("→ Column 0 and 1 share i=0, columns 2 and 3 share i=1, etc.")

    # Calculate 10000^(2i/d_model)
    div_term = 10000 ** (div_indices / d_model)

    print(f"\ndiv_term = 10000 ** (div_indices / {d_model})")
    print(f"Values: {div_term}")
    print("\nBreaking it down:")
    for idx, val in zip(div_indices, div_term):
        exponent = idx / d_model
        print(f"  i={int(idx)}: 10000^({idx}/{d_model}) = 10000^{exponent:.2f} = {val:.2f}")

    print("\n→ Low indices = small divisor = high frequency (changes fast)")
    print("→ High indices = large divisor = low frequency (changes slow)")

    # =========================================================================
    # STEP 3: Broadcasting - the magic step
    # =========================================================================
    print("\n" + "─" * 70)
    print("STEP 3: Broadcasting (position / div_term)")
    print("─" * 70)

    scaled_pos = position / div_term

    print(f"\nposition shape: {position.shape}")
    print(f"div_term shape: {div_term.shape}")
    print(f"scaled_pos = position / div_term")
    print(f"Result shape: {scaled_pos.shape}")

    print("\nHow broadcasting works:")
    print("  position (4,1):     div_term (2,):")
    print("  [[0],               [1.0, 100.0]")
    print("   [1],       /")
    print("   [2],")
    print("   [3]]")
    print("\n  NumPy 'stretches' both to (4,2) and divides element-wise:")

    print(f"\nscaled_pos =\n{scaled_pos}")
    print("\n→ Each row: position divided by each wavelength")
    print("→ This is the 'angle' we'll pass to sin/cos")

    # =========================================================================
    # STEP 4: Apply sin and cos
    # =========================================================================
    print("\n" + "─" * 70)
    print("STEP 4: Apply sin (even cols) and cos (odd cols)")
    print("─" * 70)

    pe = np.zeros((seq_len, d_model), dtype=np.float32)

    # Even columns get sin
    pe[:, 0::2] = np.sin(scaled_pos)
    # Odd columns get cos
    pe[:, 1::2] = np.cos(scaled_pos)

    print("\npe[:, 0::2] = np.sin(scaled_pos)  # columns 0, 2, 4, ...")
    print("pe[:, 1::2] = np.cos(scaled_pos)  # columns 1, 3, 5, ...")

    print(f"\nsin(scaled_pos) =\n{np.sin(scaled_pos)}")
    print(f"\ncos(scaled_pos) =\n{np.cos(scaled_pos)}")

    print("\nInterleaving sin/cos into final matrix:")
    print(f"\npe =\n{pe}")

    # =========================================================================
    # STEP 5: Verify against expected values
    # =========================================================================
    print("\n" + "─" * 70)
    print("STEP 5: Verify the results")
    print("─" * 70)

    print("\nLet's manually verify position 0:")
    print("  sin(0) = 0, cos(0) = 1")
    print(f"  Row 0 should be [0, 1, 0, 1]: {pe[0]}")

    print("\nLet's verify position 1, dimension 0:")
    print(f"  sin(1 / 10000^(0/{d_model})) = sin(1 / 1) = sin(1) = {np.sin(1):.4f}")
    print(f"  pe[1, 0] = {pe[1, 0]:.4f}")

    return pe


def visualize_frequencies(seq_len: int = 50, d_model: int = 8):
    """Show how different dimensions oscillate at different frequencies."""
    print("\n" + "=" * 70)
    print("VISUALIZING FREQUENCIES")
    print("=" * 70)

    position = np.arange(seq_len, dtype=np.float32)[:, np.newaxis]
    div_indices = np.arange(0, d_model, 2, dtype=np.float32)
    div_term = 10000 ** (div_indices / d_model)
    scaled_pos = position / div_term

    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(scaled_pos)
    pe[:, 1::2] = np.cos(scaled_pos)

    print(f"\nseq_len={seq_len}, d_model={d_model}")
    print("\nDimension 0 (fastest frequency) - first 20 positions:")
    print("pos: ", end="")
    for i in range(20):
        print(f"{i:5}", end="")
    print()
    print("val: ", end="")
    for i in range(20):
        val = pe[i, 0]
        print(f"{val:5.2f}", end="")
    print()

    print(f"\nDimension {d_model-2} (slowest frequency) - first 20 positions:")
    print("pos: ", end="")
    for i in range(20):
        print(f"{i:5}", end="")
    print()
    print("val: ", end="")
    for i in range(20):
        val = pe[i, d_model - 2]
        print(f"{val:5.2f}", end="")
    print()

    print("\n→ Notice: Dim 0 oscillates rapidly, Dim", d_model - 2, "barely changes")
    print("→ This multi-scale pattern uniquely identifies each position!")

    # ASCII visualization
    print("\n\nASCII Plot - Dimension 0 (sin, fast):")
    for i in range(min(20, seq_len)):
        val = pe[i, 0]
        bar_pos = int((val + 1) * 20)  # Scale [-1,1] to [0,40]
        bar = " " * bar_pos + "●"
        print(f"pos {i:2}: [{bar:40}] {val:+.3f}")

    print(f"\nASCII Plot - Dimension {d_model-2} (sin, slow):")
    for i in range(min(20, seq_len)):
        val = pe[i, d_model - 2]
        bar_pos = int((val + 1) * 20)
        bar = " " * bar_pos + "●"
        print(f"pos {i:2}: [{bar:40}] {val:+.3f}")


def show_uniqueness(seq_len: int = 8, d_model: int = 4):
    """Demonstrate that each position has a unique encoding."""
    print("\n" + "=" * 70)
    print("UNIQUENESS: Each position has a unique fingerprint")
    print("=" * 70)

    position = np.arange(seq_len, dtype=np.float32)[:, np.newaxis]
    div_indices = np.arange(0, d_model, 2, dtype=np.float32)
    div_term = 10000 ** (div_indices / d_model)
    scaled_pos = position / div_term

    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(scaled_pos)
    pe[:, 1::2] = np.cos(scaled_pos)

    print(f"\nEncoding matrix ({seq_len} positions × {d_model} dimensions):\n")

    # Header
    print("     ", end="")
    for d in range(d_model):
        print(f"  dim{d} ", end="")
    print()

    # Values
    for pos in range(seq_len):
        print(f"pos{pos} ", end="")
        for d in range(d_model):
            print(f" {pe[pos, d]:+.3f}", end="")
        print()

    # Distance matrix
    print("\n\nPairwise Euclidean distances:")
    print("     ", end="")
    for j in range(seq_len):
        print(f" pos{j} ", end="")
    print()

    for i in range(seq_len):
        print(f"pos{i} ", end="")
        for j in range(seq_len):
            dist = np.linalg.norm(pe[i] - pe[j])
            if i == j:
                print("   -  ", end="")
            else:
                print(f" {dist:.3f}", end="")
        print()

    print("\n→ All off-diagonal distances are non-zero = unique positions!")


def interactive_explorer():
    """Let user explore different configurations."""
    print("\n" + "=" * 70)
    print("TRY DIFFERENT VALUES")
    print("=" * 70)

    configs = [
        (4, 4, "Minimal example from the prompt"),
        (8, 6, "Slightly larger"),
        (16, 8, "Typical small transformer"),
    ]

    for seq_len, d_model, desc in configs:
        print(f"\n>>> {desc}: seq_len={seq_len}, d_model={d_model}")

        position = np.arange(seq_len, dtype=np.float32)[:, np.newaxis]
        div_indices = np.arange(0, d_model, 2, dtype=np.float32)
        div_term = 10000 ** (div_indices / d_model)
        scaled_pos = position / div_term

        pe = np.zeros((seq_len, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(scaled_pos)
        pe[:, 1::2] = np.cos(scaled_pos)

        print(f"Shape: {pe.shape}")
        print(f"First row (pos=0): {pe[0]}")
        print(f"Last row (pos={seq_len-1}): {pe[-1]}")


def main():
    print("\n" + "█" * 70)
    print("  POSITIONAL ENCODING - INTERACTIVE EXPLORATION")
    print("█" * 70)

    # Step by step walkthrough
    step_by_step_positional_encoding(seq_len=4, d_model=4)

    # Visualize frequencies
    visualize_frequencies(seq_len=50, d_model=8)

    # Show uniqueness
    show_uniqueness(seq_len=6, d_model=4)

    # Try different configs
    interactive_explorer()

    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print("""
1. POSITION VECTOR: Shape (seq_len, 1) - one value per position

2. DIVISION TERM: Controls wavelength
   - Small div_term (low dims) = high frequency = changes rapidly
   - Large div_term (high dims) = low frequency = changes slowly

3. BROADCASTING: NumPy automatically aligns shapes
   - (seq_len, 1) / (d_model/2,) → (seq_len, d_model/2)

4. SIN/COS INTERLEAVING:
   - Even columns (0, 2, 4...): sin
   - Odd columns (1, 3, 5...): cos
   - This gives each position a unique "fingerprint"

5. WHY IT WORKS:
   - Multi-scale frequencies = unique encoding for each position
   - Relative positions have consistent relationships
   - No learned parameters = generalizes to any sequence length
""")


if __name__ == "__main__":
    main()
