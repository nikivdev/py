"""MLX Training - Complete training examples.

This module shows how to train models with MLX,
including compiled training loops for maximum performance.

Run: python -m mlx.training
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from functools import partial
import time


def simple_regression():
    """Simple linear regression example."""
    print("=== Simple Linear Regression ===\n")

    # Generate data: y = 2x + 1 + noise
    mx.random.seed(42)
    x = mx.random.uniform(shape=(100, 1))
    y = 2 * x + 1 + mx.random.normal(shape=(100, 1)) * 0.1

    # Model
    model = nn.Linear(1, 1)

    # Loss function
    def loss_fn(model, x, y):
        pred = model(x)
        return mx.mean((pred - y) ** 2)

    # Optimizer
    optimizer = optim.SGD(learning_rate=0.1)

    # Training
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for epoch in range(100):
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}, Loss: {loss.item():.6f}")

    # Check learned parameters
    w = model.weight.item()
    b = model.bias.item()
    print(f"\nLearned: y = {w:.3f}x + {b:.3f}")
    print(f"Target:  y = 2.000x + 1.000")


def mlp_classification():
    """MLP classification on synthetic data."""
    print("\n=== MLP Classification ===\n")

    # Generate spiral dataset
    mx.random.seed(42)
    n_samples = 200
    n_classes = 3

    def make_spiral(n_samples, n_classes):
        xs, ys = [], []
        for c in range(n_classes):
            t = mx.linspace(0, 4 * 3.14159, n_samples) + mx.random.uniform(shape=(n_samples,)) * 0.2
            r = t / (4 * 3.14159)
            theta = t + c * 2 * 3.14159 / n_classes
            x = mx.stack([r * mx.cos(theta), r * mx.sin(theta)], axis=1)
            xs.append(x)
            ys.append(mx.full((n_samples,), c, dtype=mx.int32))
        return mx.concatenate(xs), mx.concatenate(ys)

    x_train, y_train = make_spiral(n_samples, n_classes)
    print(f"Data shape: {x_train.shape}, Labels shape: {y_train.shape}")

    # MLP model
    class MLP(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.layers = [
                nn.Linear(in_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, out_dim),
            ]

        def __call__(self, x):
            for layer in self.layers[:-1]:
                x = nn.relu(layer(x))
            return self.layers[-1](x)

    model = MLP(2, 64, n_classes)

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

    n_params = count_params(model.parameters())
    print(f"Model parameters: {n_params}")

    # Loss and optimizer
    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y).mean()

    optimizer = optim.Adam(learning_rate=0.01)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training loop
    for epoch in range(200):
        loss, grads = loss_and_grad(model, x_train, y_train)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if epoch % 40 == 0:
            # Accuracy
            logits = model(x_train)
            preds = mx.argmax(logits, axis=1)
            acc = mx.mean(preds == y_train).item()
            print(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}, Acc: {acc:.3f}")


def compiled_training():
    """Compiled training loop for maximum performance."""
    print("\n=== Compiled Training ===\n")

    # Simple model for benchmarking
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)

        def __call__(self, x):
            x = nn.relu(self.fc1(x))
            x = nn.relu(self.fc2(x))
            return self.fc3(x)

    model = SimpleNet()
    optimizer = optim.Adam(learning_rate=0.001)

    # Dummy MNIST-like data
    x = mx.random.uniform(shape=(256, 784))
    y = mx.random.randint(0, 10, shape=(256,))

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y).mean()

    # === Non-compiled training ===
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Warmup
    for _ in range(10):
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    # Time non-compiled
    start = time.perf_counter()
    for _ in range(100):
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
    non_compiled_time = time.perf_counter() - start

    print(f"Non-compiled: {non_compiled_time*10:.2f}ms per step")

    # === Compiled training ===
    # Reset model
    model = SimpleNet()
    optimizer = optim.Adam(learning_rate=0.001)

    # Capture state for compilation
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(x, y):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, x, y)
        optimizer.update(model, grads)
        return loss

    # Warmup (includes compilation)
    for _ in range(10):
        loss = train_step(x, y)
        mx.eval(state)

    # Time compiled
    start = time.perf_counter()
    for _ in range(100):
        loss = train_step(x, y)
        mx.eval(state)
    compiled_time = time.perf_counter() - start

    print(f"Compiled:     {compiled_time*10:.2f}ms per step")
    print(f"Speedup:      {non_compiled_time/compiled_time:.2f}x")


def transformer_example():
    """Simple transformer block example."""
    print("\n=== Transformer Block ===\n")

    class TransformerBlock(nn.Module):
        def __init__(self, dim, num_heads, mlp_ratio=4.0):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiHeadAttention(dim, num_heads)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, int(dim * mlp_ratio)),
                nn.GELU(),
                nn.Linear(int(dim * mlp_ratio), dim),
            )

        def __call__(self, x, mask=None):
            x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask=mask)
            x = x + self.mlp(self.norm2(x))
            return x

    # Create block
    dim = 256
    num_heads = 8
    block = TransformerBlock(dim, num_heads)

    # Input: (batch, seq_len, dim)
    x = mx.random.uniform(shape=(4, 32, dim))

    # Forward pass
    y = block(x)
    mx.eval(y)

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

    n_params = count_params(block.parameters())
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {n_params:,}")

    # Benchmark
    mx.eval(block(x))  # Warmup

    start = time.perf_counter()
    for _ in range(100):
        mx.eval(block(x))
    elapsed = time.perf_counter() - start

    print(f"Time per forward: {elapsed*10:.3f}ms")


def main():
    """Run all training examples."""
    simple_regression()
    mlp_classification()
    compiled_training()
    transformer_example()


if __name__ == "__main__":
    main()
