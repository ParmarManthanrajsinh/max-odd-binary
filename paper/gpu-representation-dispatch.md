# Representation-Driven Dispatch on GPUs

**How GPU frameworks select algorithms, fuse kernels, and specialize code
based on data properties — a survey of the techniques hiding inside every
major GPU computing stack.**

---

## Abstract

The previous paper in this series (*Representation-Driven Dispatch: One
Idea, Many Names*) identified a pattern that recurs across computer
science: selecting a more efficient implementation based on observable
properties of the data. On GPUs, the stakes are higher. A wrong choice
doesn't cost 2x — it can cost 100x, because GPU performance is dominated
by memory bandwidth, occupancy, and launch overhead, all of which are
acutely sensitive to data layout, element type, and problem shape.

This paper surveys how representation-driven dispatch manifests in GPU
computing. We identify five distinct layers at which it operates — from
hand-tuned CUDA primitives up through ML compiler graphs — and show that
the same "observe, select, execute" pattern appears at every level, with
increasing degrees of automation and abstraction.

---

## 1. Why GPUs Amplify the Problem

On a CPU, choosing a suboptimal algorithm might cost 2-5x. On a GPU, the
penalties compound:

**Memory hierarchy sensitivity.** A GPU has ~20 MB of SRAM (shared
memory / registers) at ~19 TB/s, but ~40-80 GB of HBM at ~1-3 TB/s. An
algorithm that fits in SRAM is 10,000x faster per byte than one that
spills to HBM. Representation-driven dispatch on GPUs is often really
about *fitting in the right memory level*.

**Launch overhead.** Each CUDA kernel launch costs 5-15 microseconds.
For small operations (< 100 us), launch overhead dominates. Fusing N
operations into one kernel eliminates N-1 launches — but only if the
operations are *fusible*, which depends on their types and data flow.

**Occupancy cliffs.** GPU occupancy depends on registers per thread,
shared memory per block, and thread count. A kernel parameterized for
fp32 may achieve 75% occupancy; the same kernel for fp64 (double the
register pressure) may drop to 37%. Type-aware dispatch isn't optional
— it's the difference between using half the chip and all of it.

**Warp divergence.** GPUs execute 32 threads in lockstep (a warp). If
data properties cause threads within a warp to take different branches,
both paths execute serially. Representation-aware algorithms avoid this
by choosing branchless strategies when data properties permit.

These factors make GPUs the environment where representation-driven
dispatch matters most — and where the GPU ecosystem has developed the
most sophisticated instances of it.

---

## 2. The Five Layers

We organize GPU representation-driven dispatch into five layers, from
lowest to highest abstraction:

| Layer | What dispatches | Property observed | When |
|-------|-----------------|-------------------|------|
| **Primitive libraries** | Hand-tuned CUDA kernels | Element type, problem size, sparsity pattern | Compile time (templates) or runtime |
| **Iterator/fusion abstractions** | Lazy expression trees | Operation category (pointwise, reduction, scan) | Expression construction time |
| **Tensor compilers** | Generated kernels | Tensor format, shape, hardware target | Compile/install time |
| **ML graph compilers** | Fused subgraphs | Op type, dtype, shape, device | JIT trace time |
| **Auto-schedulers** | Schedule parameters | Empirical measurement | Tuning time |

---

## 3. Layer 1: Primitive Libraries — CUB, CUTLASS, cuSPARSE

At the lowest level, CUDA libraries select algorithms based on type and
size through C++ template specialization and runtime branching.

### CUB: Radix Sort as a Case Study

NVIDIA's CUB library is the most explicit example. Its `DeviceRadixSort`
has a multi-level dispatch system:

**Type-based dispatch.** The sort kernel is templated on key type. For
8-bit keys (`uint8`), CUB uses a single-pass counting sort. For 32-bit
keys, it uses a 4-pass radix sort. For 64-bit keys, 8 passes. The number
of passes, the histogram bin count, and the scan strategy all depend on
`sizeof(KeyT)`.

**Size-based dispatch.** For small arrays (< ~30K elements), CUB selects
a single-block sort that fits entirely in shared memory. For large arrays,
it uses a multi-block decomposition with global memory communication.

**Architecture-based dispatch.** CUB's "tuning infrastructure" selects
thread counts, items-per-thread, and block-level primitives based on the
GPU's compute capability. A sort tuned for an A100 (SM80) uses different
parameters than one for an RTX 4090 (SM89). Per-type tuning for uint8
and uint16 keys achieved 18% median speedup for keys and 35% for
key-value pairs.

**Policy objects.** CUB encapsulates these decisions in *policy* types
that are selected at compile time via template specialization:

```cpp
template <typename KeyT, typename ValueT, int ARCH>
struct DeviceRadixSortPolicy;

// Specialization for SM80, 32-bit keys
template <>
struct DeviceRadixSortPolicy<uint32_t, NullType, 800> {
  using OnesweepPolicy = AgentRadixSortOnesweepPolicy<
    256,        // BLOCK_THREADS
    21,         // ITEMS_PER_THREAD
    DominantT,  // computed from KeyT
    RADIX_BITS_8,
    BLOCK_SCAN_WARP_SCANS
  >;
};
```

This is the same pattern as CBQN's `grade.h` decision tree, but resolved
at compile time through C++ templates rather than at runtime through
type-tag branches.

### CUTLASS: GEMM Dispatch

NVIDIA's CUTLASS library for matrix multiplication takes type-based
dispatch to an extreme. A single GEMM call dispatches based on:

- **Input types:** fp16, bf16, tf32, fp32, fp64, int8, fp8, fp4
- **Accumulator type:** often wider than inputs (fp16 inputs → fp32 accum)
- **Tile shape:** 128×128×64, 64×64×32, etc. — chosen per type and arch
- **Epilogue:** what happens after the GEMM (bias add, ReLU, scaling)
- **Hardware instruction:** Tensor Core MMA on Ampere vs. Hopper's TMA

Each combination yields a distinct kernel. CUTLASS 3.x on Hopper has
hundreds of kernel variants. The dispatch key is a tuple of
`(input_type, output_type, accumulator_type, tile_shape, epilogue, arch)`.

The epilogue fusion is particularly relevant: rather than writing the
GEMM result to global memory and launching a separate kernel for bias
addition or activation, CUTLASS fuses the epilogue into the GEMM kernel's
writeback phase. The property that enables this is knowing the epilogue's
*type* at compile time — it must be a pointwise function over the output
tile.

### cuSPARSE: Format-Driven Dispatch

cuSPARSE exemplifies dispatch based on *data layout* rather than element
type. The same operation (sparse matrix-vector multiply) has radically
different implementations for:

| Format | Layout | Best for | Strategy |
|--------|--------|----------|----------|
| CSR    | Compressed rows | General sparse | Row-parallel |
| COO    | Coordinate pairs | Very sparse | Atomic adds |
| BSR    | Block sparse | Structured blocks | Block-dense GEMM |
| ELL    | Fixed-width | Regular sparsity | Coalesced vectorized |

The user specifies the format, but the library selects the kernel. Recent
versions also support *algorithm selection* within a format — for CSR
SpMV, cuSPARSE can choose between merge-path, row-split, and
warp-per-row strategies based on the matrix's row-length distribution.

---

## 4. Layer 2: Iterator Composition and Lazy Fusion — Thrust

Thrust (now part of NVIDIA's CCCL) achieves kernel fusion through a
compile-time mechanism: **iterator adaptors** that compose operations
without materializing intermediates.

### The Transform-Iterator Pattern

```cpp
auto squares = thrust::make_transform_iterator(
    data.begin(),
    [] __device__ (float x) { return x * x; }
);
// No kernel launched yet — squares is a lazy iterator

thrust::reduce(squares, squares + n);
// ONE kernel: reads data, squares each element, reduces
```

The key insight: `transform_iterator` doesn't allocate memory or launch a
kernel. It wraps another iterator with a functor. When an algorithm like
`reduce` dereferences the iterator, it calls the functor inline. The
compiler sees through the abstraction and generates a single kernel that
reads, transforms, and reduces in one pass.

### Composition Stacks

Iterators compose:

```cpp
auto expr = make_transform_iterator(
    make_zip_iterator(make_tuple(x.begin(), y.begin())),
    [] __device__ (auto t) {
        auto [a, b] = t;
        return sqrt(a*a + b*b);
    }
);
// Still lazy — no memory allocated, no kernel launched
thrust::sort(expr, expr + n);
// ONE kernel: reads x and y, computes sqrt(a²+b²), sorts
```

The "representation" being dispatched on is the *category of the
operation*. Pointwise operations become iterator adaptors (lazy, fusible).
Structural operations (sort, reduce, scan) are the materialization
boundaries that trigger kernel launches.

This is exactly the lazy/eager boundary that defines fusion in every
system we'll examine — the question is always: *which operations are
fusible (lazy) and which force materialization (eager)?*

### The Modern Evolution: cuda::zip_transform_iterator

CCCL's newer `cuda::zip_transform_iterator` combines zip and transform
into a single adaptor, avoiding intermediate tuple materialization. This
is a micro-optimization, but it illustrates the principle: even within
the iterator abstraction, reducing intermediate representations improves
performance.

---

## 5. Layer 3: Tensor Compilers — TACO, Halide, Triton

Tensor compilers generate GPU kernels from high-level specifications,
with the compilation process itself driven by properties of the data
representation.

### TACO: Format as a First-Class Dispatch Key

The Tensor Algebra Compiler (Kjolstad et al., 2017) is the purest
example of representation-driven code generation. In TACO, the *format*
of each tensor dimension is an explicit part of the computation
specification:

```cpp
// Same computation, different generated code
C(i,j) = A(i,k) * B(k,j)

// If A is CSR and B is dense: row-parallel SpMM
// If A is COO and B is dense: sorted merge + scatter
// If A is dense and B is dense: tiled GEMM
// If A is ELL and B is CSR: yet another strategy
```

TACO's compiler uses "iteration graphs" and "merge lattices" to generate
loop nests that are correct and efficient for arbitrary combinations of
sparse and dense formats. The 2018 format abstraction paper (Chou,
Kjolstad, Amarasinghe) showed that computing directly on COO format
without converting to CSR can be up to 3.6x faster — the format *is* the
algorithm selector.

### Halide: Schedule as Dispatch Policy

Halide separates the algorithm (what to compute) from the schedule (how
to compute it). For GPU targets, the schedule specifies:

- Tile sizes for GPU thread blocks
- Which loops become GPU blocks vs. threads
- Shared memory staging decisions
- Vectorization width within threads

The schedule is the "dispatch policy" — it maps the same algorithm to
radically different GPU code based on properties like image dimensions,
filter sizes, and target GPU. Halide's auto-scheduler (Adams et al., 2019)
uses a learned cost model to select schedules, making it an instance of
ML-driven algorithm selection.

### Triton: Autotuned Tile Dispatch

OpenAI's Triton exposes GPU kernel writing at a higher level than CUDA,
with explicit autotuning support:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4),
    ],
    key=['M', 'N', 'K'],  # re-tune when these change
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...):
    ...
```

The `key` parameter is the property observed. The `configs` are the
portfolio. `triton.autotune` empirically selects the fastest
configuration *per problem shape*, caching the result. This is FFTW's
planning strategy applied to GPU tile sizes.

Recent Triton development (2024-2025) adds **warp specialization** — the
autotuner can also select how to partition work across warp groups within
a kernel, based on the balance between compute and memory access in the
specific operation.

---

## 6. Layer 4: ML Graph Compilers — XLA, TorchInductor

Machine learning frameworks operate on computation graphs where nodes are
tensor operations. The graph compiler's job is to decide which nodes to
*fuse* into single GPU kernels and which to keep separate. This decision
is representation-driven dispatch at the graph level.

### XLA: Fusion Categories

XLA (used by JAX and TensorFlow) classifies operations into fusion
categories and uses these categories to determine what can be merged:

| Category | Description | Fusible with |
|----------|-------------|--------------|
| **kLoop** | Element-wise ops (add, mul, exp) | Other kLoop ops |
| **kInput** | Ops reading input in a pattern (transpose, broadcast) | Limited |
| **kOutput** | Reduction epilogues | The preceding reduction |

XLA's fusion pass walks the graph and greedily merges adjacent nodes
whose categories are compatible. A chain of element-wise operations
(add → mul → exp → sigmoid) becomes a single kLoop kernel. A reduction
followed by an element-wise op becomes a reduction with a fused epilogue.

The property observed is the **operation category** — not the data values,
but the *structural type* of the computation. This is a compile-time
dispatch that happens during graph lowering, analogous to C++ tag dispatch
but on a dataflow graph rather than iterator types.

### JAX: Tracing as Specialization

JAX's `jit` transformation traces a Python function with abstract values
(shapes and dtypes, not concrete data) to produce an XLA HLO graph. This
is literally partial evaluation: the "static input" is the shape/dtype
metadata, and the "residual program" is the compiled GPU kernel.

```python
@jax.jit
def f(x):
    return jnp.sum(jnp.exp(x))

# First call with shape (1024,) fp32:
#   → traces, compiles, caches: one fused exp+sum kernel
# Second call with shape (1024,) fp32:
#   → cache hit, no recompilation
# Call with shape (2048,) fp32:
#   → re-traces, compiles new kernel for new shape
```

The dispatch key is `(function_identity, input_shapes, input_dtypes)`.
Each unique key gets a specialized compiled kernel.

### TorchInductor: Triton Code Generation

PyTorch 2.0's TorchInductor compiler traces the computation graph via
`torch.compile` and generates Triton kernels. Its fusion decisions are
driven by operation categories:

- **Pointwise ops** → fused into single Triton kernels
- **Reductions** → separate kernels with potential epilogue fusion
- **Matmuls** → dispatched to cuBLAS or CUTLASS

The Inductor scheduler groups fusible nodes and generates one Triton
kernel per group. Within each kernel, the code generator selects block
sizes and reduction strategies based on tensor shapes and dtypes.

A current limitation (as of 2024-2025): fusion of pointwise ops *into*
reductions ("tiled point-wise and reduction fusion") requires careful
dimension alignment. If the pointwise op's shape doesn't tile-align with
the reduction's iteration space, fusion is blocked. This illustrates how
representation properties (shape divisibility) gate optimization
decisions.

### FlashAttention: Manual Representation-Aware Fusion

FlashAttention (Dao, 2022) is the most celebrated example of
representation-aware GPU kernel design. Standard attention computes:

```
softmax(Q @ K^T / sqrt(d)) @ V
```

Naive implementation materializes the N×N attention matrix in HBM.
FlashAttention observes that:

1. The attention matrix is too large for SRAM (N² × sizeof(float))
2. But the *tiles* of Q, K, V fit in SRAM
3. Softmax can be computed in an *online* fashion (tile by tile) using a
   running max and sum correction

The "representation property" is the *memory hierarchy fit*: knowing that
Q/K/V tiles fit in SRAM while the full attention matrix doesn't drives
the selection of a tiled, online algorithm over the naive materialization.

This reduced memory from O(N²) to O(N), enabling 3x speedup on GPT-2
and 5-20x memory reduction. It's the GPU analog of CBQN's boolean sort
optimization: recognizing a structural property of the data and selecting
a fundamentally different algorithm.

---

## 7. Layer 5: Auto-Schedulers and Learned Dispatch

At the highest level, the dispatch decision itself is automated through
search or machine learning.

### TVM / Ansor: Auto-Scheduling

Apache TVM's Ansor auto-scheduler (Zheng et al., 2020) automatically
constructs a search space of GPU kernel implementations and finds the
best one empirically:

1. **Analyze** the tensor computation's DAG structure
2. **Generate** candidate schedules (tile sizes, loop orderings,
   vectorization, thread binding)
3. **Measure** each candidate on real hardware
4. **Select** the fastest, cache for future use

The property observed is the *empirical runtime* on specific hardware —
the most expensive but most accurate form of dispatch. Ansor achieves
competitive or superior performance to hand-tuned kernels across diverse
workloads.

A 2024 improvement introduced Dynamic Gradient Descent search, achieving
93.7% of full Ansor performance in 1 hour instead of 6 hours for BERT
models — reducing the cost of the dispatch decision itself.

### Learned Cost Models

Several systems now use neural networks to predict the best
implementation without exhaustive measurement:

- **Halide's auto-scheduler** (Adams et al., 2019) uses a learned cost
  model to predict schedule performance from features like loop nest
  structure and memory access patterns.
- **TGraph** uses learned models for tensor layout optimization, improving
  prediction accuracy from 29.8% to 67.4%.
- **Neptune** (2024) uses algebraic analysis to find fusible reduction
  patterns that previous heuristic-based systems miss, achieving 1.35x
  speedup over Triton and TVM on attention benchmarks.

---

## 8. ArrayFire and Futhark: GPU Array Languages

Two systems bridge the gap between array language semantics and GPU
execution, applying representation-driven dispatch in ways most similar to
CBQN.

### ArrayFire: JIT Fusion via AST

ArrayFire is a GPU array library that uses lazy evaluation and JIT
compilation:

1. Element-wise operations build an **Abstract Syntax Tree** (AST) at
   runtime rather than launching kernels immediately
2. When a non-fusible operation (reduction, sort) or explicit `eval()` is
   encountered, the AST is compiled into a single CUDA/OpenCL kernel via
   NVRTC
3. The fused kernel executes all queued element-wise operations in one
   pass

The expression `sqrt(x*x + y*y) < 1` generates one kernel, not five.
The dispatch decision is binary: is this operation *element-wise*
(defer) or *structural* (materialize)?

ArrayFire's JIT compilation cost (~600ms per unique expression) is the
price of runtime dispatch. This is analogous to a JIT inline cache's
first-call overhead, amortized over subsequent executions.

### Futhark: Compiler-Level Fusion for GPUs

Futhark (Henriksen et al., 2017) is a purely functional array language
that compiles to GPU code. Its approach to fusion is unique:

**Moderate flattening.** Rather than aggressively flattening all nested
parallelism (as NESL did), Futhark "moderately" flattens — exploiting
enough parallelism to saturate the GPU while preserving the data access
patterns that enable optimization. The property observed is the *nesting
structure* of the parallel computation.

**Incremental flattening** (Henriksen et al., 2019) goes further:
generating *multiple kernel versions* for different degrees of available
parallelism, with runtime selection based on the actual input sizes. This
is classic algorithm selection — a portfolio of kernels, dispatched by
input shape at runtime.

**Fusion rules.** Futhark's second-order array combinators (map, reduce,
scan, filter) have algebraic fusion rules:

```
map f . map g  →  map (f . g)        -- producer-consumer fusion
map f △ map g  →  map (f △ g)        -- horizontal fusion
reduce f . map g  →  reduce-map f g  -- map-reduce fusion
```

These rules are applied by the compiler during optimization, driven by
the *types* of the combinators (element-wise, reduction, scan). This is
the functional programming analog of XLA's kLoop/kInput/kOutput
categorization.

---

## 9. cuDF: Type Dispatch in GPU DataFrames

RAPIDS cuDF applies representation-driven dispatch to tabular data on
GPUs. Each column has a runtime dtype, and operations dispatch based on
it:

**Compiled dispatch.** For common operations (binary ops, aggregations),
cuDF pre-compiles kernels for each supported type pair and dispatches at
runtime via type-indexed function tables. This mirrors CBQN's approach
exactly — a type tag selects from a pre-built kernel portfolio.

**JIT dispatch.** For complex expressions (user-defined transforms, string
operations), cuDF JIT-compiles CUDA kernels via NVRTC at runtime,
specializing on the column's actual dtype. The first execution pays ~600ms
compilation cost; subsequent calls with the same types hit a cache.

**Constrained type dispatchers.** A 2024 optimization reduced compilation
time by 85% (from 712s to 110s) by constraining which type combinations
are pre-compiled, based on analysis of which combinations actually occur
in practice. This is meta-dispatch: selecting *which dispatch paths to
compile* based on usage patterns.

---

## 10. The Fusion Boundary Problem

Across all layers, the central question of GPU representation-driven
dispatch is: **where to place the materialization boundary?**

Every GPU system must decide which operations to fuse into a single kernel
and where to break the fusion chain. This decision is driven by operation
properties:

| Operation type | Typically fusible? | Why |
|----------------|-------------------|-----|
| Element-wise (map) | Yes | No cross-element communication |
| Broadcast | Yes | Predictable access pattern |
| Reshape/transpose | Sometimes | May require data movement |
| Reduction | Boundary | Requires cross-thread synchronization |
| Sort | Boundary | Global data movement |
| Scatter/gather | Boundary | Irregular access pattern |
| Scan (prefix sum) | Boundary | Sequential dependency |
| GEMM | Boundary | Calls optimized subroutine |

The "fusibility" of an operation is a property of its *category* — its
structural type in the dataflow graph. This categorization is the GPU
equivalent of CBQN's element type tag: a small integer that gates a
major algorithmic decision.

### The Lazy/Eager Spectrum

Systems differ in how they implement the boundary:

| System | Lazy operations | Eager boundaries | Mechanism |
|--------|----------------|------------------|-----------|
| Thrust | transform_iterator | sort, reduce, scan | C++ templates |
| ArrayFire | Element-wise AST | reduce, sort, eval() | Runtime JIT |
| XLA | kLoop fusion | kOutput reduction | Graph compiler |
| TorchInductor | Pointwise nodes | Reduction, matmul | Graph scheduler |
| Futhark | map, zip | reduce, scan, scatter | Compiler fusion rules |

Every system converges on the same answer: **pointwise operations are
lazy; structural operations are eager.** The differences are in mechanism
(compile-time vs. runtime), granularity (per-operation vs. per-subgraph),
and how aggressively they push the boundary (can reductions be partially
fused? can scans?).

---

## 11. Connecting to the CPU Story

The GPU landscape mirrors the CPU taxonomy from the previous paper, but
with amplified consequences:

| CPU concept | GPU analog | Amplification factor |
|-------------|-----------|---------------------|
| Type squeezing (CBQN) | dtype-driven kernel selection (CUTLASS) | Register pressure → occupancy cliffs |
| Counting sort for small types | CUB's type-specialized radix sort | Shared memory capacity limits |
| Fusion via deforestation | Kernel fusion via lazy iterators/graphs | Memory bandwidth is 10-100x more precious |
| Algorithm selection by size | Tile size autotuning | Grid/block dimensions interact non-linearly |
| Packed booleans | Format-driven dispatch (cuSPARSE) | Irregular access → warp divergence |

The fundamental insight transfers directly: **the more you know about the
data representation, the better algorithm you can select.** On GPUs, the
performance gap between the best and worst choice is larger, the decision
space is more complex (types × shapes × tile sizes × architectures), and
the payoff for getting it right is greater.

---

## 12. Open Problems

### Automatic Format Selection for Sparse Computation

cuSPARSE requires users to specify the sparse format. No production system
automatically selects between CSR, COO, BSR, and ELL based on the
sparsity pattern. TACO's format abstraction provides the theoretical
framework, but practical auto-selection on GPUs remains open.

### Cross-Kernel Dispatch

Current systems optimize within a single kernel or a single fusion group.
Optimizing the *sequence* of kernel launches — for example, choosing
whether to fuse ops A+B or B+C when you can't fuse all three — is a
combinatorial problem. Korch (2024) formalizes this as binary linear
programming but doesn't scale to large graphs.

### Dynamic Representation Changes

Most GPU dispatch is static (compile-time) or one-shot (first execution).
Few systems re-dispatch mid-computation when data properties change — for
example, switching from a dense kernel to a sparse kernel when a matrix
becomes sparse during iterative computation. Database adaptive query
execution does this for CPU; the GPU equivalent is largely unexplored.

### Learned Dispatch Policies

Can a neural network learn the dispatch policy itself? Current learned
cost models predict performance of individual kernels, but predicting the
best *sequence* of dispatch decisions across an entire program remains
open.

---

## 13. Bibliography

### Primitive Libraries

1. Merrill, D. (2015). "CUB: Cooperative primitives for CUDA." NVIDIA.

2. Thakkar, V. et al. (2023). "CUTLASS: Fast linear algebra in CUDA C++."
   NVIDIA.

3. NVIDIA. (2024). "cuSPARSE Library." CUDA Toolkit Documentation.

### Iterator Composition

4. Bell, N. and Hoberock, J. (2012). "Thrust: A productivity-oriented
   library for CUDA." *GPU Computing Gems, Jade Edition*, 359-371.

### Tensor Compilers

5. Kjolstad, F. et al. (2017). "The tensor algebra compiler." *OOPSLA*,
   77:1-29.

6. Chou, S., Kjolstad, F., and Amarasinghe, S. (2018). "Format abstraction
   for sparse tensor algebra compilers." *OOPSLA*, 123:1-30.

7. Ragan-Kelley, J. et al. (2013). "Halide: A language and compiler for
   optimizing parallelism, locality, and recomputation in image processing
   pipelines." *PLDI*, 519-530.

8. Tillet, P., Kung, H.T., and Cox, D. (2019). "Triton: An intermediate
   language and compiler for tiled neural network computations." *MAPL*.

### ML Graph Compilers

9. XLA Team. (2017). "XLA: Optimizing compiler for machine learning."
   Google/OpenXLA.

10. Ansel, J. et al. (2024). "PyTorch 2: Faster machine learning through
    dynamic Python bytecode transformation and graph compilation." *ASPLOS*.

11. Dao, T. et al. (2022). "FlashAttention: Fast and memory-efficient exact
    attention with IO-awareness." *NeurIPS*.

12. Zhao, Z. et al. (2024). "Neptune: Advanced ML operator fusion for
    locality and parallelism on GPUs." *arXiv:2510.08726*.

13. Zheng, L. et al. (2024). "Korch: Optimal kernel orchestration for
    tensor programs." *arXiv:2406.09465*.

### Auto-Schedulers

14. Zheng, L. et al. (2020). "Ansor: Generating high-performance tensor
    programs for deep learning." *OSDI*, 863-879.

15. Adams, A. et al. (2019). "Learning to optimize Halide with tree
    search and random programs." *ACM TOG*, 38(4).

### GPU Array Languages

16. Malcolm, J. et al. (2012). "ArrayFire: A GPU acceleration platform."
    *SPIE Defense + Security*.

17. Henriksen, T. et al. (2017). "Futhark: Purely functional GPU
    programming with nested parallelism and in-place array updates."
    *PLDI*, 556-571.

18. Henriksen, T. et al. (2019). "Incremental flattening for nested data
    parallelism." *PPoPP*, 53-64.

### GPU DataFrames

19. RAPIDS Team. (2024). "cuDF: GPU DataFrame library." NVIDIA.

---

*This is the second paper in a series. The first,
"Representation-Driven Dispatch: One Idea, Many Names," surveys the
concept across all of computer science. This paper focuses on its
manifestation in GPU computing, where the performance consequences are
most dramatic.*
