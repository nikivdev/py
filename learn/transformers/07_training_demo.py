#!/usr/bin/env python3
"""
Step 7: Training Demo - How Transformers Learn

The magic: W_Q, W_K, W_V, and FFN weights are all LEARNED via backpropagation.
The model learns HOW to pay attention to minimize prediction error.

This demo trains a tiny transformer on a simple pattern.

Run: uv run python 07_training_demo.py
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import math


# ============================================================
# Mini Transformer (simplified for training demo)
# ============================================================


class MiniTransformer(nn.Module):
    """
    Tiny decoder-only transformer for learning demo.
    Task: Learn to copy input sequence (identity function).
    """

    def __init__(self, vocab_size: int, d_model: int = 32, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_pos_encoding(64, d_model)

        self.layers = []
        for _ in range(num_layers):
            self.layers.append({
                "attn": nn.MultiHeadAttention(d_model, num_heads),
                "ffn1": nn.Linear(d_model, d_model * 4),
                "ffn2": nn.Linear(d_model * 4, d_model),
                "norm1": nn.LayerNorm(d_model),
                "norm2": nn.LayerNorm(d_model),
            })

        self.output = nn.Linear(d_model, vocab_size)

    def _create_pos_encoding(self, max_len, d_model):
        position = mx.arange(max_len)[:, None]
        dim = mx.arange(0, d_model, 2)
        div_term = mx.exp(dim * (-math.log(10000.0) / d_model))
        pe_even = mx.sin(position * div_term)
        pe_odd = mx.cos(position * div_term)
        pe = mx.concatenate([pe_even[:, :, None], pe_odd[:, :, None]], axis=2)
        return pe.reshape(max_len, d_model)

    def __call__(self, x):
        seq_len = x.shape[1]

        # Embed and add positional encoding
        h = self.embedding(x) * math.sqrt(self.d_model)
        h = h + self.pos_encoding[:seq_len]

        # Causal mask
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)

        # Transformer layers
        for layer in self.layers:
            # Self-attention with residual
            normed = layer["norm1"](h)
            attn_out = layer["attn"](normed, normed, normed, mask=mask)
            h = h + attn_out

            # FFN with residual
            normed = layer["norm2"](h)
            ffn_out = layer["ffn2"](nn.gelu(layer["ffn1"](normed)))
            h = h + ffn_out

        return self.output(h)


def cross_entropy_loss(logits, targets):
    """Cross-entropy loss for language modeling."""
    vocab_size = logits.shape[-1]

    # Flatten for loss computation
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    # Log softmax + negative log likelihood
    log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)

    # Gather the log probabilities for the correct tokens
    batch_indices = mx.arange(targets_flat.shape[0])
    target_log_probs = log_probs[batch_indices, targets_flat]

    return -mx.mean(target_log_probs)


def demo():
    print("=" * 70)
    print("TRAINING DEMO: HOW TRANSFORMERS LEARN")
    print("=" * 70)

    # Simple task: Learn to copy input (shifted by 1)
    # Input:  [1, 2, 3, 4, 5]
    # Target: [2, 3, 4, 5, 6]  (predict next token)
    vocab_size = 20
    seq_len = 8
    batch_size = 16
    num_epochs = 100

    mx.random.seed(42)

    print(f"""
Task: Next token prediction on simple sequences

  Input:  [1, 2, 3, 4, 5, 6, 7, 8]
  Target: [2, 3, 4, 5, 6, 7, 8, 9]

The model must learn to predict: next_token = current_token + 1

This requires learning:
  1. Positional information (where am I?)
  2. Token identity (what token is here?)
  3. The +1 pattern (simple, but requires learning!)
""")

    # Create model
    model = MiniTransformer(vocab_size, d_model=32, num_heads=4, num_layers=2)

    # Count parameters
    def count_params(params):
        total = 0
        if isinstance(params, dict):
            for v in params.values():
                total += count_params(v)
        elif isinstance(params, list):
            for v in params:
                total += count_params(v)
        elif hasattr(params, 'size'):
            total += params.size
        return total

    num_params = count_params(model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = optim.Adam(learning_rate=0.001)

    # Loss and grad function
    def loss_fn(model, x, y):
        logits = model(x)
        return cross_entropy_loss(logits, y)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    print("\n" + "-" * 70)
    print("TRAINING")
    print("-" * 70)

    losses = []
    for epoch in range(num_epochs):
        # Generate batch: sequences of consecutive numbers
        start = mx.random.randint(1, vocab_size - seq_len - 1, (batch_size,))
        x = mx.stack([mx.arange(s, s + seq_len) for s in start.tolist()])
        y = x + 1  # Target is next token

        # Forward and backward pass
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        losses.append(loss.item())

        if epoch % 20 == 0 or epoch == num_epochs - 1:
            print(f"  Epoch {epoch:3d}: Loss = {loss.item():.4f}")

    print("\n" + "-" * 70)
    print("TESTING THE TRAINED MODEL")
    print("-" * 70)

    # Test on a new sequence
    test_input = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    expected = [2, 3, 4, 5, 6, 7, 8, 9]

    logits = model(test_input)
    predictions = mx.argmax(logits, axis=-1)[0].tolist()

    print(f"\n  Input:      {test_input[0].tolist()}")
    print(f"  Expected:   {expected}")
    print(f"  Predicted:  {predictions}")

    correct = sum(1 for p, e in zip(predictions, expected) if p == e)
    print(f"  Accuracy:   {correct}/{len(expected)} = {correct/len(expected)*100:.0f}%")

    print("\n" + "-" * 70)
    print("WHAT THE MODEL LEARNED")
    print("-" * 70)

    print("""
Through backpropagation, the model learned to set its weights so that:

  1. ATTENTION WEIGHTS (W_Q, W_K, W_V):
     - Query: "What position am I at?"
     - Key: "Here's what I know about each position"
     - The attention pattern likely learns to focus on the current position

  2. FFN WEIGHTS (W1, W2):
     - Maps token embedding to "token + 1" embedding
     - Stores the "+1" transformation in its weights

  3. OUTPUT PROJECTION:
     - Converts final hidden state to vocabulary distribution
     - High probability for correct next token

This is a toy example, but REAL transformers learn similarly:
  - Instead of "+1", they learn complex language patterns
  - "After 'the', nouns are likely"
  - "After 'Paris is the capital of', 'France' is likely"
  - The same mechanism, just learned from billions of tokens!
""")

    # Show loss curve
    print("-" * 70)
    print("LOSS CURVE (ASCII)")
    print("-" * 70)

    max_loss = max(losses)
    height = 10
    width = 50

    # Downsample losses if needed
    step = max(1, len(losses) // width)
    sampled_losses = [losses[i] for i in range(0, len(losses), step)]

    print()
    for row in range(height, 0, -1):
        threshold = (row / height) * max_loss
        line = "  │"
        for loss in sampled_losses:
            if loss >= threshold:
                line += "█"
            else:
                line += " "
        if row == height:
            line += f"  {max_loss:.2f}"
        elif row == 1:
            line += f"  {min(losses):.2f}"
        print(line)

    print("  └" + "─" * len(sampled_losses) + "►")
    print("   0" + " " * (len(sampled_losses) - 5) + f"{num_epochs}")
    print("                    Epochs")

    print("\n" + "=" * 70)
    print("THE MAGIC OF LEARNING ATTENTION")
    print("=" * 70)

    print("""
The remarkable insight:

  The model doesn't have hardcoded rules about language.
  ALL knowledge comes from adjusting weights to minimize loss.

  W_Q, W_K, W_V learn:
    - What features to look for
    - What features to advertise
    - What information to pass along

  FFN weights learn:
    - Facts about the world
    - Transformations between concepts
    - Complex non-linear patterns

  Everything emerges from: ∂Loss/∂weights → update weights

This is why Transformers are so powerful:
  - Architecture provides the CAPACITY to learn relationships
  - Training provides the KNOWLEDGE
  - Scale (data + parameters) determines the QUALITY
""")


if __name__ == "__main__":
    demo()
