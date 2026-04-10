Here's the deep dive. The results aren't crazy at all — they make perfect sense once you see what CBQN is actually doing under the hood. There are several compounding advantages.

---

## 1. The Bit-Packing Superpower

The single most important thing CBQN does is **represent booleans as packed bit arrays** — 1 bit per element, not 1 byte. A 1000-element boolean array is **125 bytes** (16 u64 words), not 1000 bytes.

Every operation on booleans then works on **64 elements per machine word** — a free 64x parallelism multiplier before SIMD even enters the picture.

C++ and Rust work with `std::string` / `Vec<u8>` — **1 byte per character** throughout. They never get this density advantage.

---

## 2. Boolean Sort is O(n/64), Not O(n)

This is the big one for the "sort+rotate alt" solution (`"01"⊏˜1⌽·∨'1'⊸=`, 159ns).

When CBQN sorts a boolean array, it doesn't compare-and-swap. It does this:

```230:248:/home/cph/CBQN/src/builtins/grade.h
// Sort descending on booleans:
  if (xe==el_bit) {
    u64* xp = bitany_ptr(x);
    u64* rp; r = m_bitarrc(&rp, x);
    usz sum = bit_sum(xp, n);        // count 1s: ~16 POPC instructions
    u64 n0 = GRADE_UD(n-sum, sum);   // number of leading values
    // ... fill with all-1 words, boundary word, all-0 words
  }
```

That's it. **Count 1s with ~16 hardware `POPC` instructions, then fill ~16 u64 words.** For n=1000, this is roughly 32 instructions total. The `1⌽` rotate is then just two `memcpy` calls on ~125 bytes.

Compare to C++ `std::ranges::partition` which must touch each of 1000 bytes individually with branching swaps (617-668ns).

---

## 3. Even the Raw Character Sort is O(n) Counting Sort

For the direct `1⌽∨` solution (277ns, sorting the c8 string directly), CBQN stores "01"-only strings as **c8 arrays** (1 byte per char). At n=1000 (>= 256), the sort hits `COUNTING_SORT_i8`:

```260:261:/home/cph/CBQN/src/builtins/grade.h
      } else {
        COUNTING_SORT_i8;  // n >= 256: O(n) counting sort for 1-byte data
```

With SIMD (`SINGELI_AVX2`), the histogram is built by `simd_count_i8` — vectorized counting into 256 bins — then the output is written via a `⌈`-scan or dense fill. For 2-value data ('0' and '1'), the "sparse" path fires and uses a SIMD prefix-max scan to fill the sorted output. This is fundamentally O(n) with a small constant.

C++ `std::sort` on a string is O(n log n) comparison-based. Even `std::partition` is O(n) but with per-byte branching and pointer chasing.

---

## 4. SIMD Pipeline for Count Solutions

The tacit count solution (`"101"/˜(≠(1∾˜(1-˜⊢)∾-)(+´'1'⊸=))`, 151ns) executes this pipeline:

| Step | Operation    | CBQN Implementation                         | Work for n=1000          |
| ---- | ------------ | ------------------------------------------- | ------------------------ |
| 1    | `'1'⊸=`      | Singeli SIMD: vector compare + pack to bits | ~4 vector ops on c8 data |
| 2    | `+´` on bits | `bit_sum`: hardware `POPC` per u64 word     | ~16 POPC instructions    |
| 3    | Arithmetic   | Three scalar i64 ops                        | 3 instructions           |
| 4    | `"101"/˜`    | Singeli `rep_const`: SIMD broadcast + store | ~32 vector stores        |

The comparison step is in `cmp.singeli`:

```62:80:/home/cph/CBQN/src/singeli/src/cmp.singeli
// Vector compare all lanes, pack results into bits via hom_to_int + store_bits
// One pass, no intermediate allocation of a byte array
```

The scalar is broadcast to a SIMD register, compared against loaded chunks of the character array, and the comparison results are packed directly into a bit array via `hom_to_int` (movemask on x86) + `store_bits`. No intermediate byte-sized boolean array ever exists.

`bit_sum` then uses hardware popcount:

```411:416:/home/cph/CBQN/src/builtins/fold.c
NOINLINE i64 bit_sum(u64* x, u64 am) {
  i64 r = 0;
  for (u64 i = 0; i < (am>>6); i++) r+= POPC(x[i]);
  if (am&63) r+= POPC(x[am>>6]<<(64-am & 63));
  return r;
}
```

The replicate step (`"101"/˜⟨n-1, total-n, 1⟩`) uses Singeli's constant-replicate with SIMD broadcast: splat '1' to a vector, store n-1 copies; splat '0', store the rest; append '1'. This is essentially **two memsets + one byte store**.

Total work: ~1000 bytes read, ~1000 bytes written, ~50 instructions of real computation. It's approaching the speed of a single `memcpy`.

---

## 5. Why C++/Rust Can't Match This

The Rust count+construct (349ns) does:

```rust
let n = s.iter().filter(|&&c| c == b'1').count();  // byte-at-a-time count
let mut r = Vec::with_capacity(s.len());
r.extend(std::iter::repeat(b'1').take(n - 1));     // memset (good)
r.extend(std::iter::repeat(b'0').take(s.len() - n)); // memset (good)
r.push(b'1');
```

The counting step iterates byte-by-byte (the compiler *may* auto-vectorize the filter+count, but it's not guaranteed to match hand-tuned SIMD). The memset parts are competitive, but:

- **No bit-packing**: the count loop touches 1000 bytes vs. CBQN touching 125 bytes of bits
- **Allocation overhead**: `Vec::with_capacity` calls the allocator; CBQN's typed array allocator is a tuned bump/buddy allocator
- **No fused compare-to-bits**: Rust counts through a byte iterator; CBQN fuses compare + pack + popcount

The C++ count+construct (554-699ns) uses `std::ranges::count` + string construction, which has similar limitations plus `std::string` SSO/allocation overhead.

---

## Summary: CBQN's Architectural Advantages

1. **Bit-packed booleans**: 64 elements per word = automatic 8-64x less memory traffic
2. **Type-specialized algorithms**: Boolean sort = popcount + memset. Character sort = O(n) counting sort. Not a generic comparison sort.
3. **Singeli SIMD code generator**: Hand-tuned vector code for every primitive, compiled to native AVX2/SSE/NEON — not relying on auto-vectorization
4. **Fused operations**: Compare-to-bits is one pass, not "compare to bytes, then pack"
5. **Custom allocator**: Buddy allocator tuned for array workloads, avoiding `malloc` overhead
6. **Algorithm selection by data type and size**: The `grade.h` code has different paths for n<16, n<256, n<32768, etc., each optimal for that range

The BQN code *looks* like it's doing more work (sort an array! rotate!), but CBQN's runtime recognizes that sorting booleans is just counting, and every step maps to a few dozen SIMD instructions on data that fits in L1 cache.