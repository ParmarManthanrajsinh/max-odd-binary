# Representation-Driven Dispatch: One Idea, Many Names

**How choosing the right algorithm based on data properties appears across
compiler theory, generic programming, JIT compilation, databases, array
languages, autotuning, and adaptive algorithms — and why no one community
owns the concept.**

---

## Abstract

A single idea recurs across nearly every branch of computer science:
*given knowledge about the properties of the data, select a more efficient
implementation.* This idea has been independently discovered, named, and
formalized in at least a dozen communities. A compiler calls it **strength
reduction**. A C++ programmer calls it **tag dispatching**. A database calls
it **access path selection**. A JIT compiler calls it **type feedback**. An
autotuning library calls it **plan selection**. An array language runtime
calls it **type specialization**.

These are all the same thing.

This document traces the idea through its many incarnations, identifies the
common structure, and argues that the fragmentation of terminology has
obscured what is arguably one of the most important principles in practical
computer science: *algorithms should be chosen by the machine, not the
programmer, based on what is known about the data at the point of
execution.*

---

---

## 1. The Core Pattern

Every instance of this idea shares a common structure:

```
 ┌─────────────────────────────────────────────────────────┐
 │  INPUT: a computation to perform + some observable      │
 │         property of the operands                        │
 │                                                         │
 │  DECISION: select from a portfolio of implementations   │
 │            based on that property                       │
 │                                                         │
 │  OUTPUT: the same semantic result, faster               │
 └─────────────────────────────────────────────────────────┘
```

The "observable property" varies:

| Property observed         | Who observes it       | When              |
|---------------------------|-----------------------|-------------------|
| Static type               | Compiler              | Compile time      |
| Value range / bit width   | Array runtime         | Array creation    |
| Iterator category         | C++ template system   | Compile time      |
| Concrete class            | JIT compiler          | First call        |
| Table statistics          | Query optimizer       | Query planning    |
| Hardware features          | Loader / resolver     | Program startup   |
| Input distribution        | Adaptive algorithm    | During execution  |
| Empirical runtime         | Autotuner             | Offline / install |

But the structure is always the same: *observe, select, execute.*

---

## 2. Compiler Theory: Specialization and Strength Reduction

### Strength Reduction (Classical)

The oldest form. Compilers have replaced expensive operations with cheaper
equivalent ones since the 1960s:

- `x * 2` → `x << 1`
- `x / 4` → `x >> 2` (for unsigned x)
- `x % 8` → `x & 7` (for unsigned x)

The enabling **property** is the compile-time constant value of one operand.
The compiler *knows* that the divisor is a power of two, so it selects a
shift instead of a division.

This is exactly the same pattern as CBQN selecting `popcount + memset`
instead of comparison sort when it knows the array is boolean. The only
difference is what property is observed and when.

### Type Specialization

Modern compilers specialize code paths based on type:

- Integer arithmetic vs. floating-point arithmetic
- Signed vs. unsigned comparison sequences
- Pointer-width operations vs. narrow operations

In languages with unboxed types, this happens at compile time. In languages
with tagged types (Lisp, Smalltalk, array languages), it happens at
runtime.

### Names in this community

- Strength reduction
- Peephole optimization
- Type specialization
- Operator lowering
- Idiom recognition

---

## 3. Partial Evaluation: The Theoretical Apex

Partial evaluation is the *general theory* of which all forms of
specialization are instances.

Given a program `P(static_input, dynamic_input)`, a partial evaluator
produces a *residual program* `P_s(dynamic_input)` that is specialized to
the known static input. The residual program produces the same results as
the original but is typically much faster because decisions that depended
only on `static_input` have been resolved.

The key insight, formalized by Futamura (1971) and systematized by Jones,
Gomard, and Sestoft (1993), is that **compilation itself is partial
evaluation**: an interpreter specialized on a particular source program
becomes a compiled program.

### The Futamura Projections

1. Specializing an interpreter on a program → compiled program
2. Specializing a specializer on an interpreter → compiler
3. Specializing a specializer on itself → compiler generator

Every system in this paper can be understood as a form of partial
evaluation where the "static input" is some property of the data:

| System                    | Static input              | Dynamic input       |
|---------------------------|---------------------------|---------------------|
| Compiler strength red.    | Constant operand value    | Other operand       |
| C++ tag dispatch          | Iterator category type    | Container contents  |
| CBQN sort                 | Element type tag          | Array values        |
| Database optimizer        | Table statistics          | Query predicates    |
| FFTW planner              | Transform size            | Input data          |
| JIT inline cache          | Receiver class            | Message arguments   |

### Key references

- Futamura, Y. (1971). "Partial evaluation of computation process — an
  approach to a compiler-compiler." *Systems, Computers, Controls*, 2(5),
  45-50.
- Jones, N.D., Gomard, C.K., and Sestoft, P. (1993). *Partial Evaluation
  and Automatic Program Generation.* Prentice Hall.

---

## 4. Generic Programming: Concepts and Tag Dispatch

Alexander Stepanov's vision for the STL was that algorithms should be
parameterized by *concepts* — abstract requirements on types — and that
different concepts should enable different implementations.

### Tag Dispatching (C++98)

The STL uses empty "tag types" to select algorithm variants at compile
time:

```cpp
struct input_iterator_tag {};
struct forward_iterator_tag : input_iterator_tag {};
struct bidirectional_iterator_tag : forward_iterator_tag {};
struct random_access_iterator_tag : bidirectional_iterator_tag {};

// std::advance selects O(1) or O(n) based on iterator category
template<class It, class N>
void advance(It& it, N n, random_access_iterator_tag) { it += n; }

template<class It, class N>
void advance(It& it, N n, input_iterator_tag) { while(n--) ++it; }
```

The property observed is the *structural capability* of the iterator. The
same function name (`advance`) maps to fundamentally different algorithms.

### C++20 Concepts

Concepts formalize what tag dispatching did informally:

```cpp
template<std::random_access_iterator It>
void advance(It& it, ptrdiff_t n) { it += n; }

template<std::input_iterator It>
void advance(It& it, ptrdiff_t n) { while(n--) ++it; }
```

### Haskell Typeclasses

Haskell's typeclasses are the same mechanism in a different syntax:

```haskell
class Sortable a where
  sort :: [a] -> [a]

instance Sortable Int where
  sort = radixSort    -- O(n) for bounded integers

instance Sortable a => Sortable [a] where
  sort = mergeSort    -- O(n log n) comparison-based
```

The typeclass dictionary is the dispatch table. The type determines which
algorithm runs.

### Names in this community

- Tag dispatching
- Concept-based dispatch / overloading
- Typeclass resolution
- Trait-based dispatch (Rust)
- Protocol dispatch (Swift, Clojure)
- Constrained generics

### Key references

- Stepanov, A. and McJones, P. (2009). *Elements of Programming.*
  Addison-Wesley.
- Stepanov, A. and Rose, D. (2015). *From Mathematics to Generic
  Programming.* Addison-Wesley.
- Musser, D. and Stepanov, A. (1994). "Algorithm-oriented generic
  libraries." *Software: Practice and Experience*, 24(7), 623-642.

---

## 5. Object-Oriented Languages: Multiple Dispatch and Multimethods

Single dispatch (virtual methods) selects an implementation based on the
type of one object. Multiple dispatch generalizes this to select based on
the types of *all* arguments.

### CLOS (Common Lisp Object System)

CLOS (1988) was the first widely-used language with native multiple
dispatch:

```lisp
(defmethod collide ((a asteroid) (b spaceship))
  (damage-ship b))

(defmethod collide ((a spaceship) (b asteroid))
  (damage-ship a))
```

### Julia

Julia (Bezanson et al., 2017) made multiple dispatch the *central*
organizing principle of the language, combined with JIT compilation that
specializes methods on concrete argument types:

```julia
# Different sort algorithms selected by element type
sort(v::Vector{Bool}) = count_sort(v)
sort(v::Vector{UInt8}) = radix_sort(v)
sort(v::Vector{T}) where T = merge_sort(v)
```

Julia's key insight is that multiple dispatch + type inference + JIT
specialization together recover the performance of static languages while
keeping dynamic flexibility. The compiler sees `sort([true, false, true])`
and generates code that calls `count_sort` directly, with no dispatch
overhead.

### Names in this community

- Multiple dispatch / multimethods
- Dynamic dispatch
- Virtual method tables (vtables)
- Method specialization
- Predicate dispatch (more general than type-based)

### Key references

- Bobrow, D.G. et al. (1988). "Common Lisp Object System Specification."
  *ACM SIGPLAN Notices*, 23(SI), 1-142.
- Bezanson, J. et al. (2017). "Julia: A fresh approach to numerical
  computing." *SIAM Review*, 59(1), 65-98.

---

## 6. JIT Compilation: Type Feedback and Inline Caches

Dynamic languages (JavaScript, Python, Ruby, Smalltalk) don't know types
at compile time. JIT compilers recover type information at runtime through
**type feedback**: observing what types actually flow through each operation
and specializing accordingly.

### Inline Caches (Deutsch & Schiffman, 1984)

The original Smalltalk-80 optimization: at each message send site, cache
the most recent receiver class and the method it resolved to. On subsequent
calls, check if the class matches; if so, skip the lookup.

### Polymorphic Inline Caches (Hölzle, Chambers, Ungar, 1991)

Extended inline caches to store *multiple* class→method mappings per call
site, recording the actual polymorphism observed. Beyond immediate
speedup (11% median), PICs collect the type profile that enables
recompilation with speculative optimization.

### Speculative Optimization (V8, HotSpot, etc.)

Modern JIT compilers use type feedback aggressively:

1. **Profile** the types seen at each operation
2. **Speculate** that future types will match (generate optimized code for
   the common case)
3. **Deoptimize** (bail out to interpreter) if the speculation fails

V8's TurboFan compiler, for example, compiles `a + b` to a single integer
add instruction if it has only ever seen integers — but includes a guard
and deoptimization path for the case where a string appears.

This is the same pattern: observe a property (the types that actually
appear), select an implementation (integer add vs. generic add), execute.

### Names in this community

- Type feedback / type profiling
- Inline caches (ICs) / polymorphic inline caches (PICs)
- Speculative optimization / optimistic compilation
- On-stack replacement (OSR)
- Deoptimization / bailout
- Hidden classes / shapes / maps
- Tracing JIT (trace trees)

### Key references

- Deutsch, L.P. and Schiffman, A.M. (1984). "Efficient implementation of
  the Smalltalk-80 system." *POPL*, 297-302.
- Hölzle, U., Chambers, C., and Ungar, D. (1991). "Optimizing
  dynamically-typed object-oriented languages with polymorphic inline
  caches." *ECOOP*, 21-38.
- Hölzle, U. and Ungar, D. (1994). "Optimizing dynamically-dispatched
  calls with run-time type feedback." *PLDI*, 326-336.

---

## 7. Databases: Cost-Based Query Optimization

The database community has the most mature and commercially successful
version of this idea.

### Access Path Selection (Selinger et al., 1979)

The System R optimizer maintains **catalog statistics** about each table
(cardinality, number of distinct values per column, index availability) and
uses them to choose between:

- Full table scan vs. index scan
- Nested loop join vs. hash join vs. sort-merge join
- Join ordering permutations

The **cost model** estimates I/O and CPU for each plan:

```
COST = PAGE_FETCHES + w × RSI_CALLS
```

This is exactly Rice's algorithm selection problem instantiated in the
relational algebra domain: given features of the data (statistics), select
from a portfolio of algorithms (access paths, join methods) to minimize a
cost metric (estimated runtime).

### Adaptive Query Execution (Modern)

Modern engines (Spark, Presto, CockroachDB) go further: they **change the
plan mid-execution** based on runtime observations. If a hash join
encounters skewed data, it may switch to a sort-merge join. The property
is observed *during* execution, not just at planning time.

### Names in this community

- Cost-based optimization
- Access path selection
- Plan enumeration / plan space search
- Cardinality estimation
- Adaptive query processing / re-optimization
- Materialized view selection

### Key references

- Selinger, P.G. et al. (1979). "Access path selection in a relational
  database management system." *SIGMOD*, 23-34.
- Chaudhuri, S. (1998). "An overview of query optimization in relational
  systems." *PODS*, 34-43.

---

## 8. Array Languages: Runtime Type Specialization

Array languages (APL, J, BQN, NumPy) apply this principle more
aggressively than perhaps any other family of languages, because their
execution model makes it natural.

### The Key Insight

In an array language, every value is a homogeneous array with a known
element type. The runtime maintains the **tightest possible type
representation**:

- All values are 0 or 1 → packed **bit array** (1 bit per element)
- All values fit in -128..127 → **i8 array** (1 byte per element)
- All values fit in -32768..32767 → **i16 array** (2 bytes per element)
- Otherwise → **i32**, **f64**, or boxed array

This is called **type squeezing** in the APL community. It happens
automatically, invisibly to the programmer, at the boundary of every
primitive operation.

### The Payoff: Algorithm Selection Per Primitive

Every primitive checks the element type tag and dispatches to a specialized
implementation. CBQN's sort (`∨`) is the canonical example:

| Element type | n     | Algorithm selected              | Complexity   |
|-------------|-------|---------------------------------|--------------|
| bit         | any   | popcount + memset               | O(n/64)      |
| i8 / c8     | < 16  | insertion sort                  | O(n²)        |
| i8 / c8     | < 256 | radix sort (1 pass)             | O(n)         |
| i8 / c8     | ≥ 256 | counting sort (SIMD histogram)  | O(n)         |
| i16 / c16   | < 20  | insertion sort                  | O(n²)        |
| i16 / c16   | < 2¹⁵ | radix sort (2 passes)           | O(n)         |
| i16 / c16   | ≥ 2¹⁵ | counting sort                   | O(n)         |
| i32 / c32   | < 32  | insertion sort                  | O(n²)        |
| i32 / c32   | ≥ 32  | radix sort (4 passes)           | O(n)         |
| f64         | any   | comparison-based (Timsort)      | O(n log n)   |
| boxed       | any   | comparison-based (Timsort)      | O(n log n)   |

This is not 2-way dispatch. It's a **10+ way decision tree** combining
element type, array length, value range, and sortedness flags. Each leaf
is a hand-tuned implementation, often with SIMD intrinsics generated by
CBQN's Singeli compiler.

### Why This Beats C++/Rust

The fundamental asymmetry: in C++ `std::sort(s.begin(), s.end())` on a
`std::string`, the compiler knows the element type (char) but generates a
*single* generic comparison sort. It cannot know at compile time that the
string contains only '0' and '1', so it cannot select counting sort.

CBQN checks at runtime and *always* selects the optimal algorithm. The
"overhead" of the type check is a single branch on an integer tag — perhaps
1 nanosecond — but the payoff is selecting an O(n) algorithm instead of
O(n log n), or an O(n/64) algorithm instead of O(n).

### The Boolean Advantage in Detail

When CBQN evaluates `'1'=x` (compare each character to '1'), it produces
a **packed bit array**: 1000 booleans stored in 125 bytes (16 u64 words),
not 1000 bytes. Every subsequent operation gets a cascading advantage:

1. **8x less memory traffic** — 125 bytes fits in 2 cache lines
2. **64x implicit parallelism** — popcount processes 64 elements per
   instruction
3. **Algorithm upgrade** — sort becomes count+fill; sum becomes popcount

No C++ standard library function converts a `std::string` comparison
result to a `std::bitset` and then popcount-sorts it. You *could* write
this by hand, but the array language runtime does it *automatically* for
every operation, every time.

### Names in this community

- Type squeezing / type narrowing
- Representation specialization
- Internal type / storage type (APL Wiki)
- Element type dispatch
- Packed boolean optimization

### Key references

- Bernecky, R. (2016). "SIMD Boolean Array Algorithms." *Dyalog User
  Meeting.*
- Breed, L. (1966). APL\360 implementation (introduced packed boolean
  storage for APL).
- Ching, W.-M. (1986). "Program analysis and code generation in an APL/370
  compiler." *IBM Journal of Research and Development*, 30(6), 594-602.
- CBQN source: `src/builtins/grade.h`, `src/builtins/slash.c`,
  `src/singeli/src/sort.singeli`

---

## 9. Autotuning: Empirical Algorithm Selection

Some systems don't analyze properties analytically — they **measure**
which algorithm is fastest and cache the result.

### FFTW (Frigo and Johnson, 1998)

FFTW computes the Discrete Fourier Transform by:

1. Decomposing the problem into a tree of sub-problems ("codelets")
2. At install time or first use, **trying all reasonable decompositions**
   and measuring wall-clock time
3. Caching the fastest "plan" for each transform size

The property observed is the transform size + hardware characteristics. The
selection is purely empirical. FFTW consistently matches or beats
vendor-tuned FFT libraries because it explores a space that no human would
manually optimize for each hardware target.

### ATLAS (Whaley and Dongarra, 1998)

ATLAS autotunes dense linear algebra (BLAS). For matrix multiplication, it
searches over:

- Block sizes (tile sizes for cache)
- Loop orderings
- Unroll factors
- Register allocation strategies

The "property" is the hardware's cache hierarchy, pipeline depth, and
register count — measured empirically, not specified by the programmer.

### Halide (Ragan-Kelley et al., 2012)

Halide separates image processing algorithms from their *schedules*
(tile sizes, fusion decisions, vectorization, parallelism). The
schedule is a separate artifact that can be:

- Written by hand
- Auto-tuned via stochastic search
- Learned by a neural network (Adams et al., 2019)

The key contribution is making the distinction between "what to compute"
and "how to compute it" explicit and first-class.

### Names in this community

- Autotuning / self-tuning
- Plan selection / planning
- Adaptive optimization
- Empirical optimization
- Schedule search (Halide)
- Polyhedral optimization (when loop transformations are the search space)

### Key references

- Frigo, M. and Johnson, S.G. (1998). "FFTW: An adaptive software
  architecture for the FFT." *ICASSP*, 1381-1384.
- Whaley, R.C. and Dongarra, J.J. (1998). "Automatically tuned linear
  algebra software." *SC98*.
- Ragan-Kelley, J. et al. (2013). "Halide: A language and compiler for
  optimizing parallelism, locality, and recomputation in image processing
  pipelines." *PLDI*, 519-530.

---

## 10. Adaptive Algorithms: Introspection at the Algorithm Level

Some algorithms observe properties of their *own execution* and switch
strategies mid-run.

### Introsort (Musser, 1997)

Introsort begins as quicksort but monitors the recursion depth. If
partitioning is degenerating toward O(n²) behavior, it switches to
heapsort. The property observed is the depth of the recursion tree — a
proxy for input adversariality.

This became the default `std::sort` in most C++ standard libraries.

### Timsort (Peters, 2002)

Timsort scans for **natural runs** — subsequences already in order — and
merges them. On already-sorted data, it runs in O(n). On random data, it
degrades gracefully to O(n log n). The property observed is the
**presortedness** of the input.

### Pattern-Defeating Quicksort (Peters, 2021)

pdqsort detects:

- Already sorted input → linear time
- Few unique elements → Dutch national flag partitioning
- Adversarial input → heapsort fallback
- Normal input → quicksort with BlockQuicksort partitioning

This is a four-way dispatch based on runtime properties of the input
distribution, happening inside a single `sort()` call.

### Names in this community

- Adaptive algorithms
- Introspective algorithms
- Hybrid algorithms
- Input-sensitive algorithms
- Instance-optimal algorithms

### Key references

- Musser, D. (1997). "Introspective sorting and selection algorithms."
  *Software: Practice and Experience*, 27(8), 983-993.
- Peters, T. (2002). "[Python-Dev] Sorting." Python mailing list.
  (Timsort description.)
- Peters, O.R.L. (2021). "Pattern-defeating quicksort." *arXiv:2106.05123.*

---

## 11. Hardware-Level: Function Multi-Versioning and IFUNC

Even at the level of individual machine instructions, the same pattern
appears.

### glibc IFUNC

glibc compiles multiple versions of `memcpy`, `strlen`, `strcmp`, etc. —
one for SSE2, one for AVX2, one for AVX-512. At program load time, a
**resolver function** checks CPUID flags and patches the PLT entry to
point to the fastest version.

The property observed is the CPU's instruction set support. The selection
happens once at program startup.

### GCC Function Multi-Versioning

```c
__attribute__((target("avx2")))
void process(float* data, int n) { /* AVX2 version */ }

__attribute__((target("sse2")))
void process(float* data, int n) { /* SSE2 fallback */ }
```

GCC automatically generates the resolver and dispatch code.

### CBQN's Singeli

CBQN takes this further with **Singeli**, a custom DSL that generates
SIMD code for multiple architectures from a single specification. Each
primitive gets versions for scalar, SSE2, AVX2, AVX-512, and NEON,
with runtime dispatch to the best available.

### Names in this community

- Function multi-versioning (FMV)
- CPU dispatch / feature dispatch
- IFUNC / GNU indirect functions
- CPUID-based dispatch
- Micro-architecture tuning

---

## 12. Functional Programming: Rewrite Rules and Fusion

Haskell's GHC compiler applies **rewrite rules** — programmer-specified
equational transformations — to select more efficient implementations:

```haskell
{-# RULES "map/map" forall f g xs. map f (map g xs) = map (f.g) xs #-}
```

### Deforestation and Fusion

Wadler (1990) introduced **deforestation**: eliminating intermediate data
structures in compositions of list functions. Stream fusion (Coutts,
Leshchinskiy, Stewart, 2007) generalized this using rewrite rules to
fuse `map`, `filter`, `fold`, `zip`, and more into single passes.

The "property" here is the **syntactic structure** of the composition: if
the compiler can see that a producer feeds directly into a consumer, it
can fuse them. The selection is between "allocate intermediate list +
traverse twice" and "single fused loop."

### Names in this community

- Deforestation (Wadler)
- Stream fusion / shortcut fusion
- Rewrite rules
- Equational reasoning
- Build/foldr fusion

### Key references

- Wadler, P. (1990). "Deforestation: transforming programs to eliminate
  trees." *Theoretical Computer Science*, 73(2), 231-248.
- Coutts, D., Leshchinskiy, R., and Stewart, D. (2007). "Stream fusion:
  From lists to streams to nothing at all." *ICFP*, 315-326.

---

## 13. The Unifying Framework

John R. Rice formalized the general version in 1976 as the **Algorithm
Selection Problem**:

> Given a problem instance with observable features **f(x)**, select an
> algorithm **a** from a portfolio **A** that optimizes a performance
> metric **p(a, x)**.

Every system in this paper is an instantiation:

| System            | Features f(x)              | Portfolio A                  | Metric p        |
|-------------------|----------------------------|------------------------------|-----------------|
| CBQN sort         | element type, array length | 10+ sort implementations     | time            |
| C++ tag dispatch  | iterator category          | O(1) vs O(n) advance         | time complexity |
| System R          | table stats, indexes       | scan/index/join variants     | estimated I/O   |
| V8 TurboFan       | observed types at call site| specialized vs generic code  | time            |
| FFTW              | transform size, hardware   | codelet decomposition trees  | measured time   |
| Introsort         | recursion depth            | quicksort vs heapsort        | worst-case time |
| glibc IFUNC       | CPU feature flags          | SSE2/AVX2/AVX-512 versions  | throughput      |
| GHC rewrite rules | syntactic structure        | fused vs unfused code        | allocations     |

Rice's framework reveals that these communities are solving the same
optimization problem with different feature spaces, portfolios, and
metrics.

### Key reference

- Rice, J.R. (1976). "The algorithm selection problem." *Advances in
  Computers*, 15, 65-118.

---

## 14. A Taxonomy of What Is Known

The systems differ along several axes:

### When the property is observed

| Timing            | Examples                                          |
|-------------------|---------------------------------------------------|
| **Compile time**  | C++ tag dispatch, concepts, templates, Rust traits |
| **Load time**     | glibc IFUNC, Singeli CPU dispatch                  |
| **First use**     | FFTW plan caching, JIT inline caches               |
| **Every call**    | CBQN type dispatch, database adaptive execution    |
| **Mid-execution** | Introsort, Timsort, adaptive query processing      |

### What property is observed

| Property              | Examples                                        |
|-----------------------|-------------------------------------------------|
| **Type**              | Element type, iterator category, receiver class |
| **Value range**       | Type squeezing, bit-packing, counting sort      |
| **Size**              | Insertion sort for small n, radix for large n   |
| **Distribution**      | Sortedness, sparsity, skew                      |
| **Hardware**          | CPUID, cache sizes, memory bandwidth            |
| **Syntactic form**    | Rewrite rules, fusion                           |

### How the selection is made

| Method              | Examples                                         |
|---------------------|--------------------------------------------------|
| **Static dispatch** | C++ overloading, Rust traits, Haskell typeclasses|
| **Lookup table**    | CBQN type tag → function pointer array           |
| **Decision tree**   | CBQN sort (type × size branches)                 |
| **Cost model**      | Database query optimizer                          |
| **Empirical search**| FFTW, ATLAS, Halide autotuning                   |
| **Speculation**     | JIT optimistic compilation + deoptimization      |

### Whether the programmer participates

| Level                | Examples                                         |
|----------------------|--------------------------------------------------|
| **Invisible**        | CBQN type squeezing, JIT type feedback           |
| **Opt-in**           | C++ concepts, Haskell typeclasses                |
| **Manual**           | Writing FFTW wisdom files, Halide schedules      |

---

## 15. Why This Matters

### The fragmentation is harmful

Each community has developed sophisticated solutions to the algorithm
selection problem, but the solutions are largely siloed:

- Database researchers don't cite the partial evaluation literature
- Array language implementors don't cite Rice's framework
- JIT compiler papers don't reference adaptive algorithm theory
- Autotuning researchers don't connect to typeclass resolution

This means that **insights don't transfer**. CBQN's multi-level type ×
size × value-range dispatch tree is a remarkable piece of engineering, but
it was developed in isolation from the algorithm selection literature that
could have informed its design space. Conversely, the algorithm selection
community focuses on machine learning for feature extraction, while CBQN
shows that hand-crafted decision trees work beautifully when the feature
space is small and well-understood.

### The principle is becoming more important, not less

As hardware becomes more heterogeneous (CPUs with mixed core types,
GPUs, NPUs, varying SIMD widths), and as data becomes more diverse
(sparse, streaming, compressed, encrypted), the cost of choosing the
wrong algorithm increases. A single generic implementation is increasingly
far from optimal.

### What a unified theory would look like

A truly unified framework would:

1. **Characterize the feature space** — what properties of the data,
   hardware, and computation context are worth observing?
2. **Define the portfolio** — what implementations are available and what
   are their cost models?
3. **Specify the selection mechanism** — static dispatch, dynamic lookup,
   cost model, empirical search, or learned model?
4. **Bound the overhead** — what is the cost of observing the feature and
   making the selection, and when does it pay off?

Partial evaluation provides the theoretical foundation. Rice's framework
provides the problem formulation. The practical systems surveyed here
provide a rich catalog of solutions. What's missing is the synthesis.

---

## 16. Bibliography

### Foundational

1. Rice, J.R. (1976). "The algorithm selection problem." *Advances in
   Computers*, 15, 65-118.

2. Futamura, Y. (1971). "Partial evaluation of computation process — an
   approach to a compiler-compiler." *Systems, Computers, Controls*, 2(5),
   45-50.

3. Jones, N.D., Gomard, C.K., and Sestoft, P. (1993). *Partial Evaluation
   and Automatic Program Generation.* Prentice Hall.

### Generic Programming

4. Stepanov, A. and McJones, P. (2009). *Elements of Programming.*
   Addison-Wesley.

5. Musser, D. and Stepanov, A. (1994). "Algorithm-oriented generic
   libraries." *Software: Practice and Experience*, 24(7), 623-642.

6. Czarnecki, K. and Eisenecker, U. (2000). *Generative Programming:
   Methods, Tools, and Applications.* Addison-Wesley.

### JIT Compilation

7. Deutsch, L.P. and Schiffman, A.M. (1984). "Efficient implementation of
   the Smalltalk-80 system." *POPL*, 297-302.

8. Hölzle, U., Chambers, C., and Ungar, D. (1991). "Optimizing
   dynamically-typed object-oriented languages with polymorphic inline
   caches." *ECOOP*, 21-38.

### Multiple Dispatch

9. Bobrow, D.G. et al. (1988). "Common Lisp Object System Specification."
   *ACM SIGPLAN Notices*, 23(SI), 1-142.

10. Bezanson, J. et al. (2017). "Julia: A fresh approach to numerical
    computing." *SIAM Review*, 59(1), 65-98.

### Databases

11. Selinger, P.G. et al. (1979). "Access path selection in a relational
    database management system." *SIGMOD*, 23-34.

### Autotuning

12. Frigo, M. and Johnson, S.G. (1998). "FFTW: An adaptive software
    architecture for the FFT." *ICASSP*, 1381-1384.

13. Whaley, R.C. and Dongarra, J.J. (1998). "Automatically tuned linear
    algebra software." *SC98*.

14. Ragan-Kelley, J. et al. (2013). "Halide: A language and compiler for
    optimizing parallelism, locality, and recomputation in image processing
    pipelines." *PLDI*, 519-530.

### Adaptive Algorithms

15. Musser, D. (1997). "Introspective sorting and selection algorithms."
    *Software: Practice and Experience*, 27(8), 983-993.

16. Peters, T. (2002). "Timsort." Python development mailing list.

17. Peters, O.R.L. (2021). "Pattern-defeating quicksort."
    *arXiv:2106.05123.*

### Functional Programming

18. Wadler, P. (1990). "Deforestation: transforming programs to eliminate
    trees." *Theoretical Computer Science*, 73(2), 231-248.

19. Coutts, D., Leshchinskiy, R., and Stewart, D. (2007). "Stream fusion:
    From lists to streams to nothing at all." *ICFP*, 315-326.

### Array Languages

20. Bernecky, R. (2016). "SIMD Boolean Array Algorithms." *Dyalog User
    Meeting.*

21. Ching, W.-M. (1986). "Program analysis and code generation in an
    APL/370 compiler." *IBM Journal of Research and Development*, 30(6),
    594-602.

---

*This document was motivated by observing CBQN — a BQN interpreter —
outperform compiled Rust and C++ on a string sorting benchmark, not through
better hardware utilization alone, but through selecting fundamentally
different algorithms based on data properties that the compiled languages'
standard libraries ignore.*
