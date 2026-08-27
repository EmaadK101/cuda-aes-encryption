# CUDA-Accelerated AES-128 Encryption

A from-scratch AES-128 implementation, benchmarked on CPU vs. a custom CUDA kernel, to see how far GPU parallelism can push a workload that's normally thought of as sequential.

Originally built for ECEN 489 (GPU Programming & Visualization) at Texas A&M — a small, selective practicum capped at ~20 students.

## Why AES is a good fit for a GPU

AES-128 in **CTR mode** encrypts each 16-byte block independently — there's no dependency between blocks, unlike CBC mode. That independence is what makes it safely parallelizable: each GPU thread can encrypt its own block with zero risk of a race condition or cross-thread interference. That property, more than raw thread count, is the actual reason this workload maps well onto a GPU.

## What's in this repo

| File | Description |
|---|---|
| `aes_cpu.c` | Baseline CPU implementation (built on `tiny-AES-c`), single-threaded, one block at a time |
| `aes_cuda.cu` | Custom CUDA kernel — one thread per 16-byte block, full 10-round AES-128 per thread |
| `aes_key.bin` | Sample 128-bit key used for benchmarking |
| `results_*.csv` | Raw timing data, 500 runs each, for CPU/CUDA × encryption/decryption |

## How the CUDA kernel works

- **T-tables in constant memory** — `SubBytes`, `ShiftRows`, and `MixColumns` are precomputed and fused into four lookup tables (`T0`–`T3`), so each round is a handful of table lookups and XORs instead of separate byte-level operations.
- **Expanded key in shared memory** — the key schedule is expanded once per thread block (by thread 0) and shared across all threads in that block, instead of every thread redoing key expansion or hitting global memory for it.
- **One thread per block, multiple blocks per thread when needed** — each thread independently runs all 10 AES rounds for its assigned 16-byte block, keeping global memory access limited to the initial read and final write.

## Correctness, not just speed

Before trusting any of the timing numbers, the CUDA output was checked to be **bit-exact** against the CPU baseline, using the same key and IV, on the same input file. Speed means nothing if the ciphertext is wrong.

## Benchmark setup

- Test file: ~50MB plaintext (Shakespeare corpus)
- AES-128 in CTR mode, same key/IV on both implementations
- 500 iterations per implementation, per operation (encrypt/decrypt)
- Timed with high-resolution performance counters

## Results

| | CPU (avg) | CUDA (avg) | Speedup | CPU std dev | CUDA std dev |
|---|---|---|---|---|---|
| Encryption | 5.890 s | 0.0498 s | **~118×** | 0.232 s | 0.0056 s |
| Decryption | 5.522 s | 0.0490 s | **~113×** | 0.392 s | 0.0020 s |

The GPU version isn't just faster on average — it's far more *consistent*. CPU timing varied by hundreds of milliseconds run to run; CUDA varied by single-digit milliseconds.

One outlier: the very first CUDA iteration in a run consistently took longer than the rest (visible as a clear outlier in the raw data). That's consistent with one-time CUDA context initialization and memory allocation overhead on the first kernel launch — a cost that's paid once, not per-iteration.

## Takeaways

- AES-CTR's block independence is what makes it parallelizable — the algorithm choice matters as much as the hardware.
- A GPU implementation that's both faster *and* more consistent is a stronger result than raw speedup alone.
- Correctness has to be validated independently of performance — a fast wrong answer is worse than a slow right one.
