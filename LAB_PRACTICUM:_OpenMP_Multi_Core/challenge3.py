import time
import numpy as np
import numba
from numba import njit, prange

MAX_T = numba.config.NUMBA_NUM_THREADS
print(f"Numba version: {numba.__version__}, threads: {MAX_T}")

@njit(parallel=True)
def heat_step(u, u_next, alpha):
    rows, cols = u.shape
    for i in prange(1, rows - 1):
        for j in range(1, cols - 1):
            u_next[i, j] = u[i, j] + alpha * (
                u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4.0 * u[i, j]
            )

GRID_SIZE = 1500
STEPS = 300
REPEATS = 3

def make_grids(dtype):
    u = np.zeros((GRID_SIZE, GRID_SIZE), dtype=dtype)
    u[0, :] = 100.0
    u[:, 0] = 100.0
    return u, u.copy()

def run(dtype, threads):
    numba.set_num_threads(threads)
    alpha = dtype(0.20)
    best = float("inf")
    for _ in range(REPEATS):
        u, u_next = make_grids(dtype)
        start = time.perf_counter()
        for _ in range(STEPS):
            heat_step(u, u_next, alpha)
            u, u_next = u_next, u
        best = min(best, time.perf_counter() - start)
    return best

thread_counts = sorted(set(t for t in [1, 2, 4, 8, MAX_T] if t <= MAX_T))
results = {}

for dtype in (np.float64, np.float32):
    name = dtype.__name__
    # Отдельный warmup для каждого типа: Numba компилирует функцию заново
    u, u_next = make_grids(dtype)
    heat_step(u, u_next, dtype(0.20))

    print(f"\n=== {name} ===")
    print(f"{'Threads':<8} | {'Time (s)':<10} | {'Mcells/s':<10} | {'Speedup':<8}")
    print("-" * 46)
    t1 = None
    for t in thread_counts:
        elapsed = run(dtype, t)
        if t == 1:
            t1 = elapsed
        mcells = GRID_SIZE * GRID_SIZE * STEPS / elapsed / 1e6
        results[(name, t)] = (elapsed, mcells)
        print(f"{t:<8} | {elapsed:<10.3f} | {mcells:<10.1f} | {t1 / elapsed:<8.2f}x")

# Сводка для Table 1 и вопроса B
t64, m64 = results[("float64", MAX_T)]
t32, m32 = results[("float32", MAX_T)]
print("\n--- Для Table 1 (float64, max threads) ---")
print(f"Heat Diffusion Complete: {t64:.3f} s")
print(f"Throughput: {m64:.2f} Megacells/sec")
print("\n--- Для вопроса B ---")
print(f"float32 / float64 runtime factor: {t64 / t32:.2f}x faster")