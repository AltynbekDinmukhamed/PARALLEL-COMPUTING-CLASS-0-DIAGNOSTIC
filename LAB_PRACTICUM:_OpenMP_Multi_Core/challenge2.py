import time
import numpy as np
import numba
import matplotlib
matplotlib.use("Agg")  # без окна, просто сохраняем файл
import matplotlib.pyplot as plt
from numba import njit, prange

print(f"Numba version: {numba.__version__}")
print(f"Threads: {numba.config.NUMBA_NUM_THREADS}")

@njit(parallel=True)
def render_mandelbrot_rows(h, w, max_iter):
    img = np.zeros((h, w), dtype=np.int32)
    for r in prange(h):
        cy = -1.2 + (r / h) * 2.4
        for c in range(w):
            cx = -2.0 + (c / w) * 2.5
            z_real, z_imag = 0.0, 0.0
            it = 0
            while (z_real * z_real + z_imag * z_imag <= 4.0) and (it < max_iter):
                next_real = z_real * z_real - z_imag * z_imag + cx
                z_imag = 2.0 * z_real * z_imag + cy
                z_real = next_real
                it += 1
            img[r, c] = it
    return img

@njit(parallel=True)
def render_mandelbrot_cols(h, w, max_iter):
    img = np.zeros((h, w), dtype=np.int32)
    for c in prange(w):
        cx = -2.0 + (c / w) * 2.5
        for r in range(h):
            cy = -1.2 + (r / h) * 2.4
            z_real, z_imag = 0.0, 0.0
            it = 0
            while (z_real * z_real + z_imag * z_imag <= 4.0) and (it < max_iter):
                next_real = z_real * z_real - z_imag * z_imag + cx
                z_imag = 2.0 * z_real * z_imag + cy
                z_real = next_real
                it += 1
            img[r, c] = it
    return img

# JIT warmup
_ = render_mandelbrot_rows(100, 100, 50)
_ = render_mandelbrot_cols(100, 100, 50)

H, W, MAX_IT = 2500, 2500, 1000
REPEATS = 3

def bench(func):
    best = float("inf")
    out = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out = func(H, W, MAX_IT)
        best = min(best, time.perf_counter() - t0)
    return best, out

t_rows, grid_rows = bench(render_mandelbrot_rows)
t_cols, grid_cols = bench(render_mandelbrot_cols)

print(f"\nRow-Parallel Render Time:    {t_rows:.3f} s")
print(f"Column-Parallel Render Time: {t_cols:.3f} s")
print(f"Results identical: {np.array_equal(grid_rows, grid_cols)}")

# Extra: finer work chunks (closer to OpenMP dynamic scheduling)
if hasattr(numba, "set_parallel_chunksize"):
    for chunk in (1, 8, 64):
        numba.set_parallel_chunksize(chunk)
        t_chunk, _ = bench(render_mandelbrot_rows)
        print(f"Row-Parallel, chunksize={chunk:<3}:  {t_chunk:.3f} s")
    numba.set_parallel_chunksize(0)  # вернуть значение по умолчанию
else:
    print("\nset_parallel_chunksize is not available: update numba")

# Save image
plt.figure(figsize=(8, 8))
plt.imshow(grid_rows, cmap="magma", extent=[-2.0, 0.5, -1.2, 1.2])
plt.title(f"Mandelbrot {H}x{W} (Render: {t_rows:.2f}s)")
plt.axis("off")
plt.savefig("mandelbrot_output.png", dpi=300, bbox_inches="tight")
print("Saved image: mandelbrot_output.png")