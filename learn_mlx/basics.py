"""MLX Basics - Understanding the core concepts.

MLX is Apple's array framework for ML on Apple Silicon.
Key concepts: lazy evaluation, unified memory, function transforms.

Run examples: python -m mlx.basics
"""

import mlx.core as mx


def arrays_and_dtypes():
    """Arrays and data types."""
    print("=== Arrays and Data Types ===\n")

    # Create arrays - NumPy-like API
    a = mx.array([1, 2, 3, 4])
    print(f"Integer array: {a}, dtype: {a.dtype}, shape: {a.shape}")

    b = mx.array([1.0, 2.0, 3.0, 4.0])
    print(f"Float array: {b}, dtype: {b.dtype}")

    # Specific dtypes
    c = mx.array([1, 2, 3], dtype=mx.float16)
    print(f"Float16 array: {c}, dtype: {c.dtype}")

    # Random arrays
    d = mx.random.uniform(shape=(2, 3))
    print(f"Random uniform (2,3): {d}")

    e = mx.random.normal(shape=(3,))
    print(f"Random normal (3,): {e}")

    # Zeros, ones, etc.
    print(f"Zeros: {mx.zeros((2, 2))}")
    print(f"Ones: {mx.ones((2, 2))}")
    print(f"Eye: {mx.eye(3)}")


def lazy_evaluation():
    """Lazy evaluation - computations happen only when needed.

    This is one of the most important MLX concepts!
    """
    print("\n=== Lazy Evaluation ===\n")

    a = mx.array([1.0, 2.0, 3.0])
    b = mx.array([4.0, 5.0, 6.0])

    # No computation happens yet!
    c = a + b
    d = c * 2
    e = mx.exp(d)

    print("Operations defined but NOT computed yet (lazy)")
    print(f"c = a + b (lazy): shape={c.shape}, dtype={c.dtype}")

    # Computation happens when:
    # 1. mx.eval() is called
    # 2. Array is printed
    # 3. Converted to numpy
    # 4. .item() on scalar

    mx.eval(e)  # NOW computation happens
    print(f"After eval: {e}")

    # Or just print (triggers eval)
    f = mx.sin(a)
    print(f"sin(a) - eval triggered by print: {f}")


def unified_memory():
    """Unified memory - CPU and GPU share the same memory.

    No need to move data between CPU/GPU!
    """
    print("\n=== Unified Memory ===\n")

    a = mx.random.uniform(shape=(100,))
    b = mx.random.uniform(shape=(100,))

    # Both arrays live in unified memory
    # Can run on CPU or GPU without copying

    # Run on CPU
    c_cpu = mx.add(a, b, stream=mx.cpu)
    print(f"CPU result (first 5): {c_cpu[:5]}")

    # Run on GPU (same memory, no transfer needed!)
    c_gpu = mx.add(a, b, stream=mx.gpu)
    print(f"GPU result (first 5): {c_gpu[:5]}")

    # MLX automatically manages dependencies between streams
    # This is safe and efficient:
    d = mx.add(a, b, stream=mx.cpu)
    e = mx.add(d, a, stream=mx.gpu)  # Depends on CPU result
    print(f"Mixed CPU/GPU result (first 5): {e[:5]}")


def function_transforms():
    """Function transforms - grad, vmap, compile.

    These are composable! grad(grad(f)) works.
    """
    print("\n=== Function Transforms ===\n")

    # 1. Gradient computation
    print("-- Gradients --")

    def f(x):
        return mx.sum(x ** 2)

    x = mx.array([1.0, 2.0, 3.0])
    grad_f = mx.grad(f)
    print(f"f(x) = sum(x^2), x = {x}")
    print(f"grad(f)(x) = 2x = {grad_f(x)}")

    # Composable: second derivative
    def g(x):
        return mx.sin(x)

    x = mx.array(0.0)
    print(f"\ng(x) = sin(x), x = {x}")
    print(f"g(x) = {g(x)}")
    print(f"g'(x) = cos(x) = {mx.grad(g)(x)}")  # cos(0) = 1
    print(f"g''(x) = -sin(x) = {mx.grad(mx.grad(g))(x)}")  # -sin(0) = 0

    # 2. value_and_grad - get both value and gradient
    print("\n-- value_and_grad --")

    def loss_fn(w, x, y):
        pred = w * x
        return mx.mean((pred - y) ** 2)

    w = mx.array(1.0)
    x = mx.array([1.0, 2.0, 3.0])
    y = mx.array([2.0, 4.0, 6.0])

    loss_and_grad = mx.value_and_grad(loss_fn)
    loss, grad = loss_and_grad(w, x, y)
    print(f"Loss: {loss}, Gradient w.r.t w: {grad}")

    # 3. vmap - automatic vectorization
    print("\n-- vmap --")

    def add_vectors(a, b):
        return a + b

    # Batch of vectors
    xs = mx.random.uniform(shape=(4, 3))
    ys = mx.random.uniform(shape=(4, 3))

    # Vectorize over first dimension
    batched_add = mx.vmap(add_vectors)
    result = batched_add(xs, ys)
    print(f"Batched add shape: {result.shape}")


def compilation():
    """Compilation - fuse operations for speed.

    Compile merges operations and generates optimized Metal kernels.
    """
    print("\n=== Compilation ===\n")

    import math
    import time

    # GELU activation - many element-wise ops
    def gelu(x):
        return x * (1 + mx.erf(x / math.sqrt(2))) / 2

    # Compiled version
    gelu_compiled = mx.compile(gelu)

    x = mx.random.uniform(shape=(1000, 1000))

    # Warm up
    mx.eval(gelu(x))
    mx.eval(gelu_compiled(x))

    # Time regular version
    start = time.perf_counter()
    for _ in range(10):
        mx.eval(gelu(x))
    regular_time = time.perf_counter() - start

    # Time compiled version
    start = time.perf_counter()
    for _ in range(10):
        mx.eval(gelu_compiled(x))
    compiled_time = time.perf_counter() - start

    print(f"Regular GELU:  {regular_time*100:.2f}ms")
    print(f"Compiled GELU: {compiled_time*100:.2f}ms")
    print(f"Speedup: {regular_time/compiled_time:.2f}x")


def neural_network_basics():
    """Basic neural network with mlx.nn."""
    print("\n=== Neural Network Basics ===\n")

    import mlx.nn as nn
    import mlx.optimizers as optim

    # Simple MLP
    class MLP(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, out_dim)

        def __call__(self, x):
            x = nn.relu(self.fc1(x))
            return self.fc2(x)

    # Create model
    model = MLP(10, 32, 2)

    # Flatten nested parameter dict and count
    def count_params(params):
        total = 0
        for v in params.values():
            if isinstance(v, dict):
                total += count_params(v)
            else:
                total += v.size
        return total

    print(f"Model parameters: {count_params(model.parameters())}")

    # Dummy data
    x = mx.random.uniform(shape=(8, 10))
    y = mx.array([0, 1, 0, 1, 0, 1, 0, 1])

    # Loss function
    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y).mean()

    # Optimizer
    optimizer = optim.SGD(learning_rate=0.1)

    # Training step
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for i in range(5):
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        print(f"Step {i+1}, Loss: {loss.item():.4f}")


def main():
    """Run all examples."""
    arrays_and_dtypes()
    lazy_evaluation()
    unified_memory()
    function_transforms()
    compilation()
    neural_network_basics()


if __name__ == "__main__":
    main()
