#!/usr/bin/env python3
"""
Step 4: Position-wise Feed-Forward Network (FFN)

After attention, words have "talked" to each other and gathered context.
Now each word vector goes through an FFN INDIVIDUALLY.

Attention = "Look around and gather context" phase
FFN = "Think and process" phase

The FFN is where the model stores "knowledge" - facts about the world
encoded in its weights.

Run: uv run python 04_feed_forward.py
"""

import mlx.core as mx
import mlx.nn as nn


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2

    Key insight: The FFN expands to a MUCH higher dimension (4x typically),
    then compresses back. This allows complex non-linear transformations.

    d_model → d_ff (expand) → d_model (compress)
    512     → 2048          → 512

    Why expand then compress?
    - More capacity to learn complex functions
    - The hidden layer acts as a "memory bank" of features
    - ReLU activation creates sparse activations (efficiency)
    """

    def __init__(self, d_model: int, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()

        # Default: expand to 4x the model dimension
        if d_ff is None:
            d_ff = 4 * d_model

        self.d_model = d_model
        self.d_ff = d_ff

        # Two linear layers with ReLU in between
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array, verbose: bool = False) -> mx.array:
        # Step 1: Expand to higher dimension
        hidden = self.W1(x)

        if verbose:
            print(f"\n  Input shape: {x.shape}")
            print(f"  After W1 (expand to d_ff={self.d_ff}): {hidden.shape}")

        # Step 2: Non-linear activation (ReLU)
        hidden = nn.relu(hidden)

        if verbose:
            # Show sparsity - ReLU zeros out negative values
            num_active = mx.sum(hidden > 0).item()
            total = hidden.size
            sparsity = 1 - (num_active / total)
            print(f"  After ReLU: {sparsity:.1%} of activations are zero (sparse!)")

        # Step 3: Compress back to d_model
        output = self.W2(hidden)
        output = self.dropout(output)

        if verbose:
            print(f"  After W2 (compress to d_model={self.d_model}): {output.shape}")

        return output


class GeGLU_FFN(nn.Module):
    """
    Modern variant: GeGLU (Gated Linear Unit with GELU).
    Used in models like LLaMA, PaLM.

    GeGLU(x) = (x @ W_gate * GELU(x @ W_up)) @ W_down

    The "gate" learns to control information flow.
    Often performs better than simple ReLU FFN.
    """

    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        # For GeGLU, we need gate and up projections
        # Often d_ff is adjusted: (2/3) * 4 * d_model to match param count
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate = nn.gelu(self.W_gate(x))  # Gating signal
        up = self.W_up(x)  # Values to gate
        hidden = gate * up  # Element-wise gating
        return self.W_down(hidden)


def demo():
    print("=" * 60)
    print("FEED-FORWARD NETWORK (FFN)")
    print("=" * 60)

    d_model = 64
    d_ff = 256  # 4x expansion
    seq_len = 5

    mx.random.seed(42)
    x = mx.random.normal((seq_len, d_model))

    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  d_ff (hidden): {d_ff} ({d_ff // d_model}x expansion)")
    print(f"  seq_len: {seq_len}")

    ffn = FeedForward(d_model, d_ff, dropout=0.0)

    print("\n" + "-" * 50)
    print("FFN COMPUTATION (for each word individually)")
    print("-" * 50)

    output = ffn(x, verbose=True)

    print("\n" + "-" * 50)
    print("WHY THE FFN MATTERS")
    print("-" * 50)

    print("""
The FFN is like the "knowledge bank" of the transformer.

During training, the FFN weights learn to:
1. STORE factual knowledge
   - "Paris is the capital of France"
   - "Water boils at 100°C"

2. PERFORM transformations
   - Map "capital of France" → concept associated with "Paris"
   - Transform syntactic patterns into semantic understanding

3. REFINE representations
   - After attention gathered context, FFN processes it
   - Think of attention as "gather info", FFN as "make sense of it"

Key observations:
- FFN is applied INDEPENDENTLY to each position
- No interaction between positions (that's attention's job)
- But same weights W1, W2 applied to ALL positions (weight sharing)
""")

    print("-" * 50)
    print("THE ROLE OF EXPANSION")
    print("-" * 50)

    print(f"""
Why expand from {d_model} → {d_ff} then back to {d_model}?

Think of it as a "feature detector bank":

  Input (d_model={d_model})
       ↓
  [Feature Detector 1]  - "Is this about geography?"
  [Feature Detector 2]  - "Is this a question?"
  [Feature Detector 3]  - "Is this past tense?"
  ... ({d_ff} detectors)
       ↓
  ReLU - Keep only positive activations (sparse!)
       ↓
  Compress back to d_model={d_model}
  (Weighted combination of detected features)

The expansion gives capacity. The compression forces the model
to combine features meaningfully.
""")

    print("-" * 50)
    print("MODERN VARIANTS")
    print("-" * 50)

    # Create and test GeGLU variant
    geglu_ffn = GeGLU_FFN(d_model, d_ff)
    geglu_output = geglu_ffn(x)

    print(f"""
Original Transformer (2017): ReLU FFN
  FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2

Modern models use GATED variants:

1. GeGLU (LLaMA, PaLM):
   FFN(x) = (GELU(x @ W_gate) * (x @ W_up)) @ W_down

2. SwiGLU (Similar, uses Swish):
   FFN(x) = (Swish(x @ W_gate) * (x @ W_up)) @ W_down

Why gates help:
- Gate controls what information flows through
- Smoother gradients than ReLU
- Better optimization landscape

GeGLU output shape: {geglu_output.shape} (same as input!)
""")

    print("-" * 50)
    print("PARAMETER COUNT")
    print("-" * 50)

    relu_params = d_model * d_ff + d_ff + d_ff * d_model + d_model
    geglu_params = d_model * d_ff * 3  # W_gate, W_up, W_down (no bias)

    print(f"""
ReLU FFN:
  W1: {d_model} × {d_ff} + {d_ff} (bias) = {d_model * d_ff + d_ff}
  W2: {d_ff} × {d_model} + {d_model} (bias) = {d_ff * d_model + d_model}
  Total: {relu_params}

GeGLU FFN (no bias):
  W_gate: {d_model} × {d_ff} = {d_model * d_ff}
  W_up:   {d_model} × {d_ff} = {d_model * d_ff}
  W_down: {d_ff} × {d_model} = {d_ff * d_model}
  Total: {geglu_params}

Note: GeGLU has 50% more parameters for same d_ff.
Often d_ff is reduced by 2/3 to compensate.
""")


if __name__ == "__main__":
    demo()
