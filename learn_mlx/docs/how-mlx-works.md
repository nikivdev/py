# How MLX Works

MLX is Apple's ML framework for Apple Silicon. This doc explains the core concepts.

## The Big Picture

MLX has three key ideas:

1. **Lazy evaluation** - Nothing computes until you ask for results
2. **Unified memory** - CPU and GPU share the same memory (no copying)
3. **Function transforms** - Transform functions into new functions (gradients, batching, etc.)

---

## Lazy Evaluation

When you write MLX code, operations don't execute immediately. Instead, MLX builds a **compute graph**.

```python
import mlx.core as mx

a = mx.array([1.0, 2.0, 3.0])
b = mx.array([4.0, 5.0, 6.0])

c = a + b      # Nothing computed yet!
d = c * 2      # Still nothing computed!
e = mx.exp(d)  # Still nothing!

# NOW it computes (when you need the result)
mx.eval(e)     # Triggers computation of entire graph
print(e)       # This also triggers eval
```

**Why lazy?**
- MLX can optimize the whole graph before running
- Unused computations are never executed
- Memory is allocated only when needed

**When does computation happen?**
- `mx.eval(array)` - Explicit evaluation
- `print(array)` - Printing needs the values
- `array.item()` - Getting a scalar value
- Converting to numpy - `np.array(mx_array)`

---

## Function Transforms

This is the powerful part. MLX can transform functions into new functions.

### grad - Automatic Differentiation

`grad` takes a function and returns a NEW function that computes gradients.

```python
def f(x):
    return x ** 2

# f(x) = x²
# f'(x) = 2x  (derivative)

grad_f = mx.grad(f)  # Creates a NEW function

x = mx.array(3.0)
print(f(x))       # 9.0 (3² = 9)
print(grad_f(x))  # 6.0 (2 * 3 = 6)
```

**How it works internally:**
1. MLX traces `f(x)` to build the compute graph
2. Applies the chain rule backwards through the graph
3. Returns a function that computes the gradient

**Composable!** You can take gradients of gradients:

```python
def f(x):
    return mx.sin(x)

# f(x) = sin(x)
# f'(x) = cos(x)
# f''(x) = -sin(x)

x = mx.array(0.0)
print(mx.grad(f)(x))              # 1.0  (cos(0) = 1)
print(mx.grad(mx.grad(f))(x))     # 0.0  (-sin(0) = 0)
```

### value_and_grad - Get Both Output and Gradient

Often you need both the function value AND the gradient (e.g., for training):

```python
def loss(w, x, y):
    pred = w * x
    return mx.mean((pred - y) ** 2)

w = mx.array(1.0)
x = mx.array([1.0, 2.0, 3.0])
y = mx.array([2.0, 4.0, 6.0])

# Get both loss value AND gradient w.r.t. w
loss_and_grad = mx.value_and_grad(loss)
loss_val, grad_w = loss_and_grad(w, x, y)

print(f"Loss: {loss_val}")    # How wrong we are
print(f"Gradient: {grad_w}")  # Direction to improve
```

**Why not just call `loss()` then `grad(loss)()`?**
- That would compute the forward pass twice!
- `value_and_grad` shares computation, much faster

### vmap - Automatic Batching

`vmap` takes a function that works on single examples and makes it work on batches.

```python
def process_single(x):
    # Works on one vector
    return mx.sum(x ** 2)

# Without vmap: need a loop
xs = mx.random.uniform(shape=(100, 10))  # 100 vectors of size 10
results = [process_single(xs[i]) for i in range(100)]  # Slow!

# With vmap: automatic batching
batched_process = mx.vmap(process_single)
results = batched_process(xs)  # Fast! Processes all 100 at once
```

**How it works:**
1. Traces the function on a single example
2. Automatically adds batch dimension handling
3. Generates efficient batched code

**Control which axis to batch over:**

```python
def add(a, b):
    return a + b

# a has shape (batch, features)
# b has shape (features, batch)
# We want to add a[i] + b[:, i] for each i

vmap_add = mx.vmap(add, in_axes=(0, 1))  # Batch over axis 0 of a, axis 1 of b
```

### compile - Fuse Operations

`compile` optimizes a function by fusing operations into efficient GPU kernels.

```python
import math

def gelu(x):
    # GELU activation: 5 separate operations
    return x * (1 + mx.erf(x / math.sqrt(2))) / 2

# Without compile: 5 separate GPU kernel launches
result = gelu(x)

# With compile: 1 fused GPU kernel
gelu_fast = mx.compile(gelu)
result = gelu_fast(x)  # 2-5x faster!
```

**What compile does:**
1. Traces the function to capture the graph
2. Fuses compatible operations (element-wise ops combine well)
3. Generates optimized Metal shader code
4. Caches the compiled kernel

**Caveats:**
- First call is slow (compilation happens)
- Recompiles if input shapes/types change
- Use `shapeless=True` to avoid recompilation on shape changes

---

## Putting It Together: Training a Model

Here's how all the pieces work together:

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Model
model = nn.Linear(10, 1)

# Data
x = mx.random.uniform(shape=(32, 10))
y = mx.random.uniform(shape=(32, 1))

# Loss function
def loss_fn(model, x, y):
    pred = model(x)
    return mx.mean((pred - y) ** 2)

# Create gradient function
# nn.value_and_grad handles the model's parameters automatically
loss_and_grad = nn.value_and_grad(model, loss_fn)

# Optimizer
optimizer = optim.SGD(learning_rate=0.01)

# Training loop
for step in range(100):
    # Forward + backward pass (lazy - builds graph)
    loss, grads = loss_and_grad(model, x, y)

    # Update parameters (still lazy)
    optimizer.update(model, grads)

    # NOW actually compute everything
    mx.eval(model.parameters(), optimizer.state)

    print(f"Step {step}, Loss: {loss.item()}")
```

**What happens each step:**
1. `loss_and_grad()` builds a graph for forward pass + gradients (lazy)
2. `optimizer.update()` adds parameter update operations to graph (lazy)
3. `mx.eval()` executes the entire graph efficiently

---

## Quick Reference

| Transform | What it does | Example |
|-----------|--------------|---------|
| `grad(f)` | Returns gradient function | `df = mx.grad(f)` |
| `value_and_grad(f)` | Returns (value, gradient) function | `f_vg = mx.value_and_grad(f)` |
| `vmap(f)` | Adds batch dimension | `f_batch = mx.vmap(f)` |
| `compile(f)` | Optimizes/fuses operations | `f_fast = mx.compile(f)` |

**All transforms are composable:**
```python
# This works!
mx.compile(mx.vmap(mx.grad(f)))
```

---

## Mental Model

Think of MLX operations in two phases:

1. **Build phase** (lazy): Define what to compute
   - Create arrays, call operations
   - Nothing actually runs
   - MLX builds a compute graph

2. **Execute phase** (eval): Actually compute
   - Call `mx.eval()` or print/convert
   - MLX optimizes the graph
   - Runs on CPU/GPU

Function transforms work on the **graph**, not the values:
- `grad` transforms the graph to compute derivatives
- `vmap` transforms the graph to handle batches
- `compile` transforms the graph into optimized code
