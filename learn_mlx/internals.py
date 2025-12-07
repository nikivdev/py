"""MLX Internals - Understanding how MLX works under the hood.

This module explores the architecture and design of MLX.
Reference: /Users/nikiv/fork-i/ml-explore/mlx

Run: python -m mlx.internals
"""

import mlx.core as mx


def architecture_overview():
    """MLX Architecture Overview.

    MLX has several key components:

    1. CORE (C++ in mlx/):
       - array.h/cpp: The array class - a node in a compute graph
       - dtype.h/cpp: Data types (float32, float16, bfloat16, int32, etc.)
       - device.h/cpp: CPU and GPU device abstraction
       - compile.h/cpp: Graph compilation and optimization
       - fast.h/cpp: Fused operations (attention, RMS norm, etc.)

    2. BACKEND (mlx/backend/):
       - metal/: Metal GPU kernels for Apple Silicon
       - common/: CPU implementations
       - cuda/: CUDA backend (newer)

    3. PYTHON BINDINGS (python/):
       - mlx.core: Core array operations
       - mlx.nn: Neural network modules (like PyTorch nn)
       - mlx.optimizers: SGD, Adam, AdamW, etc.

    Key Design Decisions:
    - Lazy evaluation: Build compute graph, evaluate only when needed
    - Unified memory: No CPU/GPU transfers on Apple Silicon
    - Dynamic graphs: Like PyTorch, not like TensorFlow 1.x
    - Composable transforms: grad, vmap, compile can be nested
    """
    print("=== MLX Architecture ===\n")

    # The array is a node in a compute graph
    a = mx.array([1.0, 2.0, 3.0])
    b = mx.array([4.0, 5.0, 6.0])

    # These create NEW array nodes pointing to primitives
    c = a + b  # Creates Add primitive node
    d = c * 2  # Creates Multiply primitive node
    e = mx.exp(d)  # Creates Exp primitive node

    # The graph: a, b -> Add -> c -> Multiply -> d -> Exp -> e
    # Nothing computed yet!

    print("Compute graph built (not evaluated):")
    print(f"  a.shape={a.shape}, b.shape={b.shape}")
    print(f"  c = a + b")
    print(f"  d = c * 2")
    print(f"  e = exp(d)")

    # Evaluation triggers the scheduler
    mx.eval(e)
    print(f"\nAfter eval: {e}")


def memory_model():
    """Understanding MLX's memory model.

    Apple Silicon unified memory:
    - CPU and GPU share the same physical memory
    - No PCIe transfers needed (unlike NVIDIA)
    - Arrays don't "live" on CPU or GPU - they live in shared memory

    MLX allocator:
    - Uses Metal's MTLHeap for efficient allocation
    - Memory pools for common sizes
    - Automatic memory management with reference counting
    """
    print("\n=== Memory Model ===\n")

    # Create arrays - they live in unified memory
    a = mx.random.uniform(shape=(1000, 1000))

    # Check memory usage (approximate)
    size_bytes = a.size * a.itemsize()
    print(f"Array size: {a.shape}, {size_bytes / 1024:.1f} KB")

    # Operations on CPU and GPU use SAME memory
    b = mx.add(a, a, stream=mx.cpu)
    c = mx.add(a, a, stream=mx.gpu)

    # Both b and c reference the same underlying memory for 'a'
    # Only the output buffers are different
    mx.eval(b, c)
    print(f"CPU and GPU operations share input memory")

    # Memory is freed when arrays go out of scope
    # MLX uses reference counting
    del a, b, c
    print("Arrays deleted, memory returned to pool")


def lazy_eval_details():
    """Deep dive into lazy evaluation.

    When you call an MLX operation:
    1. A new array node is created
    2. The primitive (operation) is recorded
    3. Input arrays are linked as dependencies
    4. NO computation happens

    When you call mx.eval():
    1. Traverse the graph to find all needed operations
    2. Schedule operations respecting dependencies
    3. Execute on CPU/GPU
    4. Results stored in array buffers
    """
    print("\n=== Lazy Evaluation Details ===\n")

    # Build a graph
    x = mx.array([1.0, 2.0, 3.0])
    y = mx.exp(x)
    z = mx.log(y)  # Should equal x (log(exp(x)) = x)

    print("Graph: x -> exp -> y -> log -> z")
    print(f"x dtype: {x.dtype}, shape: {x.shape}")
    print(f"y dtype: {y.dtype}, shape: {y.shape} (not computed)")
    print(f"z dtype: {z.dtype}, shape: {z.shape} (not computed)")

    # Partial evaluation - only compute what's needed
    mx.eval(y)  # Computes exp(x), but NOT log(y)
    print(f"\nAfter eval(y): y = {y}")

    mx.eval(z)  # Now computes log(y)
    print(f"After eval(z): z = {z}")

    # Graph optimization happens at eval time
    # MLX can fuse operations, eliminate redundant ops, etc.


def primitives_and_transforms():
    """How primitives and transforms work.

    Primitives are the atomic operations:
    - Add, Multiply, MatMul, Exp, etc.
    - Each primitive knows how to compute forward pass
    - Each primitive knows its gradient (vjp)

    Transforms work on the graph:
    - grad: Replaces graph with gradient graph
    - vmap: Adds batch dimension handling
    - compile: Optimizes and fuses operations
    """
    print("\n=== Primitives and Transforms ===\n")

    # Simple primitive
    def f(x):
        return mx.sum(x ** 2)

    x = mx.array([1.0, 2.0, 3.0])

    # grad transform creates a NEW function
    # that computes the gradient
    grad_f = mx.grad(f)

    print("f(x) = sum(x^2)")
    print(f"f({x}) = {f(x)}")
    print(f"grad(f)({x}) = {grad_f(x)}")  # 2x = [2, 4, 6]

    # How grad works internally:
    # 1. Trace f(x) to build compute graph
    # 2. Apply chain rule backwards through graph
    # 3. Each primitive contributes its local gradient

    # vmap works similarly:
    def add_one(x):
        return x + 1

    xs = mx.array([[1, 2], [3, 4], [5, 6]])

    # vmap traces the function and adds batch handling
    batched = mx.vmap(add_one)
    print(f"\nvmap(add_one):\n{batched(xs)}")


def compilation_details():
    """How compilation works in MLX.

    mx.compile() does several things:
    1. Traces the function to capture the compute graph
    2. Applies graph optimizations:
       - Common subexpression elimination
       - Dead code elimination
       - Operation fusion (multiple ops -> one kernel)
    3. Generates optimized Metal shaders
    4. Caches the compiled kernel

    Key insights:
    - First call is slow (compilation)
    - Subsequent calls are fast (cached)
    - Recompiles if shapes/types change
    - Use shapeless=True to avoid recompilation
    """
    print("\n=== Compilation Details ===\n")

    import time

    # Function with multiple fusible operations
    def complex_fn(x):
        y = mx.exp(x)
        z = mx.sin(y)
        w = z * 2 + 1
        return mx.sum(w)

    x = mx.random.uniform(shape=(1000, 1000))

    # First compile
    compiled_fn = mx.compile(complex_fn)

    # First call - includes compilation time
    start = time.perf_counter()
    result1 = compiled_fn(x)
    mx.eval(result1)
    first_call = time.perf_counter() - start

    # Second call - uses cached kernel
    start = time.perf_counter()
    result2 = compiled_fn(x)
    mx.eval(result2)
    second_call = time.perf_counter() - start

    print(f"First call (with compilation): {first_call*1000:.2f}ms")
    print(f"Second call (cached): {second_call*1000:.2f}ms")
    print(f"Ratio: {first_call/second_call:.1f}x")

    # Shapeless compilation avoids recompilation on shape change
    shapeless_fn = mx.compile(complex_fn, shapeless=True)

    x_small = mx.random.uniform(shape=(100, 100))
    x_big = mx.random.uniform(shape=(500, 500))

    # Both use the same compiled kernel
    r1 = shapeless_fn(x_small)
    r2 = shapeless_fn(x_big)
    mx.eval(r1, r2)
    print("\nShapeless compile: same kernel for different shapes")


def stream_and_scheduling():
    """Understanding streams and scheduling.

    Streams are ordered queues of operations:
    - mx.cpu: CPU stream
    - mx.gpu: GPU stream
    - Can create custom streams

    Scheduling:
    - Operations in same stream execute in order
    - Operations in different streams can run in parallel
    - Dependencies automatically synchronized
    """
    print("\n=== Streams and Scheduling ===\n")

    a = mx.random.uniform(shape=(1000, 1000))
    b = mx.random.uniform(shape=(1000, 1000))

    # These can run in parallel (different streams, no dependency)
    c = mx.matmul(a, b, stream=mx.gpu)
    d = mx.exp(a, stream=mx.cpu)

    print("matmul on GPU, exp on CPU - can run in parallel")

    # This depends on c, so waits for GPU matmul
    e = mx.sum(c, stream=mx.cpu)

    mx.eval(d, e)
    print(f"d shape: {d.shape}, e: {e}")

    # Default stream is GPU on Apple Silicon
    default = mx.default_stream(mx.default_device())
    print(f"\nDefault device: {mx.default_device()}")


def main():
    """Run all internals examples."""
    architecture_overview()
    memory_model()
    lazy_eval_details()
    primitives_and_transforms()
    compilation_details()
    stream_and_scheduling()


if __name__ == "__main__":
    main()
