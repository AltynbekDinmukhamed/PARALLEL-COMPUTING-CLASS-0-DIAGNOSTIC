

import sys
import time
import numpy as np
from numba import njit, prange, set_num_threads

ITERATIONS = 100_000_000

# Your machine's actual cache line size (bytes), from `sysctl hw.cachelinesize`.
CACHE_LINE_BYTES = 128
INT64_BYTES = 8
STRIDE = CACHE_LINE_BYTES // INT64_BYTES  # = 16 int64 elements = exactly one cache line apart
_ONES = np.ones(64, dtype=np.int64)


# ---------------- Variant 1: Unpadded (false sharing) ----------------
@njit(parallel=True, cache=True)
def false_sharing_test(num_threads, iters, ones):
    counters = np.zeros(num_threads, dtype=np.int64)
    for tid in prange(num_threads):
        for k in range(iters):
            counters[tid] += ones[k % 64]
    return counters


# ---------------- Variant 2: Padded (cache-line isolated) ----------------
@njit(parallel=True, cache=True)
def padded_sharing_test(num_threads, iters, stride, ones):
    counters = np.zeros(num_threads * stride, dtype=np.int64)
    for tid in prange(num_threads):
        idx = tid * stride
        for k in range(iters):
            counters[idx] += ones[k % 64]
    return counters


# ---------------- Variant 3: Thread-local accumulator (Task 4.3) ----------------
@njit(parallel=True, cache=True)
def threadlocal_test(num_threads, iters, ones):
    counters = np.zeros(num_threads, dtype=np.int64)
    for tid in prange(num_threads):
        local = 0  # lives in a CPU register / local stack slot, not shared memory
        for k in range(iters):
            local += ones[k % 64]
        counters[tid] = local  # single write at the very end
    return counters


def task_scaling():
    print(f"Task 4.1 + 4.2: Unpadded vs Padded scaling (cache line = {CACHE_LINE_BYTES} bytes, stride = {STRIDE})")
    print(f"ITERATIONS per thread = {ITERATIONS:,}\n")

    # warm-up JIT
    _ = false_sharing_test(2, 1000, _ONES)
    _ = padded_sharing_test(2, 1000, STRIDE, _ONES)

    results = []
    for p in [1, 2, 4, 8, 16]:
        set_num_threads(min(p, 8))

        t0 = time.perf_counter()
        false_sharing_test(p, ITERATIONS, _ONES)
        t1 = time.perf_counter()
        time_unpadded = t1 - t0

        t2 = time.perf_counter()
        padded_sharing_test(p, ITERATIONS, STRIDE, _ONES)
        t3 = time.perf_counter()
        time_padded = t3 - t2

        ratio = time_unpadded / time_padded if time_padded > 0 else float("nan")
        results.append((p, time_unpadded, time_padded, ratio))
        print(f"P={p:2d} | unpadded={time_unpadded:.4f}s | padded={time_padded:.4f}s | ratio={ratio:.3f}x")

    with open("task4_1_2_scaling.csv", "w") as f:
        f.write("threads,unpadded_time_s,padded_time_s,speedup_ratio\n")
        for row in results:
            f.write(",".join(str(x) for x in row) + "\n")
    print("\nSaved task4_1_2_scaling.csv")


def task_threadlocal():
    print("Task 4.3: Thread-Local Accumulator variant (register-based, single write at end)")
    _ = threadlocal_test(2, 1000, _ONES)  # warm-up

    results = []
    for p in [1, 2, 4, 8, 16]:
        set_num_threads(min(p, 8))
        t0 = time.perf_counter()
        threadlocal_test(p, ITERATIONS, _ONES)
        t1 = time.perf_counter()
        results.append((p, t1 - t0))
        print(f"P={p:2d} | thread_local={t1 - t0:.4f}s")

    with open("task4_3_threadlocal.csv", "w") as f:
        f.write("threads,threadlocal_time_s\n")
        for p, t in results:
            f.write(f"{p},{t}\n")
    print("\nSaved task4_3_threadlocal.csv")
    print("Compare this file with task4_1_2_scaling.csv — thread-local should be the fastest of all three variants.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 false_sharing_lab4.py <scaling|threadlocal>")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "scaling":
        task_scaling()
    elif mode == "threadlocal":
        task_threadlocal()
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)