import time
import numpy as np
import numba
from numba import njit, prange

MAX_T = numba.config.NUMBA_NUM_THREADS
print(f"Hardware Threads Detected: {MAX_T}")

@njit(parallel=True)
def monte_carlo_pi(n_samples):
    inside_circle = 0
    for i in prange(n_samples):
        x = np.random.uniform(0.0, 1.0)
        y = np.random.uniform(0.0, 1.0)
        if x * x + y * y <= 1.0:
            inside_circle += 1
    return (4.0 * inside_circle) / n_samples

# JIT warmup: компиляция не попадает в замеры
_ = monte_carlo_pi(10_000)

SAMPLES = 120_000_000
REPEATS = 3
thread_counts = sorted(set(t for t in [1, 2, 4, 8, MAX_T] if t <= MAX_T))

results = []
t1_baseline = None

print(f"\n{'Threads':<10} | {'Time (s)':<12} | {'Speedup':<10} | {'Efficiency (%)':<15} | pi")
print("-" * 70)

for t in thread_counts:
    numba.set_num_threads(t)
    best = float("inf")
    for _ in range(REPEATS):
        start = time.perf_counter()
        pi_est = monte_carlo_pi(SAMPLES)
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)

    if t == 1:
        t1_baseline = best
    speedup = t1_baseline / best
    efficiency = speedup / t * 100.0
    results.append((t, best, speedup, efficiency))
    print(f"{t:<10} | {best:<12.4f} | {speedup:<10.2f}x | {efficiency:<15.1f} | {pi_est:.5f}")

# Сводка для Table 1
print("\n--- Для Table 1 ---")
print(f"T_1   (1 thread)  = {results[0][1]:.4f} s")
print(f"T_max ({MAX_T} threads) = {results[-1][1]:.4f} s")
print(f"Max speedup = {results[-1][2]:.2f}x, efficiency = {results[-1][3]:.1f}%")