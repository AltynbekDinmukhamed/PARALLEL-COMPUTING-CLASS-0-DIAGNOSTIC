import sys
import time
import threading
import numpy as np
from numba import njit, prange, set_num_threads

WIDTH, HEIGHT = 1920, 1080
MAX_ITER = 1000


@njit(nogil=True, cache=True)
def compute_pixel(px, py, width, height, max_iter):
    x0 = (px - width / 2.0) * 4.0 / width
    y0 = (py - height / 2.0) * 4.0 / height
    x, y = 0.0, 0.0
    iteration = 0
    while x * x + y * y <= 4.0 and iteration < max_iter:
        xtemp = x * x - y * y + x0
        y = 2.0 * x * y + y0
        x = xtemp
        iteration += 1
    return iteration


# ---------------- Static (numba prange, default even chunking) ----------------
@njit(parallel=True, cache=True)
def render_static(width, height, max_iter):
    img = np.zeros((height, width), dtype=np.int32)
    for y in prange(height):
        for x in range(width):
            img[y, x] = compute_pixel(x, y, width, height, max_iter)
    return img


# ---------------- Dynamic (manual work-queue, configurable chunk size) ----------------
@njit(nogil=True, cache=True)
def _render_row_range(img, start_row, end_row, width, height, max_iter, iter_counter, tid):
    total_iters = 0
    for y in range(start_row, end_row):
        for x in range(width):
            v = compute_pixel(x, y, width, height, max_iter)
            img[y, x] = v
            total_iters += v
    iter_counter[tid] += total_iters


def render_dynamic(width, height, max_iter, threads, chunk_size, track_imbalance=False):
    img = np.zeros((height, width), dtype=np.int32)
    iter_counter = np.zeros(threads, dtype=np.int64)  # per-thread total iterations (Task 3.4)

    next_row = [0]
    lock = threading.Lock()

    def worker(tid):
        while True:
            with lock:
                start = next_row[0]
                if start >= height:
                    return
                end = min(start + chunk_size, height)
                next_row[0] = end
            _render_row_range(img, start, end, width, height, max_iter, iter_counter, tid)

    threads_list = []
    for t in range(threads):
        th = threading.Thread(target=worker, args=(t,))
        threads_list.append(th)
        th.start()
    for th in threads_list:
        th.join()

    if track_imbalance:
        return img, iter_counter
    return img


# ---------------- Task runners ----------------
def task_static():
    print("Static scheduling baseline (numba prange, default row chunking)")
    _ = render_static(50, 50, 100)  # warm-up JIT
    t0 = time.perf_counter()
    render_static(WIDTH, HEIGHT, MAX_ITER)
    t1 = time.perf_counter()
    print(f"Static render: {t1 - t0:.4f} seconds")


def task_dynamic(threads, chunk):
    print(f"Dynamic scheduling: threads={threads}, chunk={chunk}")
    _ = render_dynamic(50, 50, 100, threads, 5)  # warm-up
    t0 = time.perf_counter()
    render_dynamic(WIDTH, HEIGHT, MAX_ITER, threads, chunk)
    t1 = time.perf_counter()
    print(f"Dynamic render (threads={threads}, chunk={chunk}): {t1 - t0:.4f} seconds")


def task_sweep():
    print("Task 3.2: Parameter Sweep Matrix — P in {2,4,8,16} x Chunk in {1,16,64,256}")
    print("3 runs per cell, this will take a while...\n")
    _ = render_dynamic(50, 50, 100, 2, 5)  # warm-up

    results = []
    for p in [2, 4, 8, 16]:
        for chunk in [1, 16, 64, 256]:
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                render_dynamic(WIDTH, HEIGHT, MAX_ITER, p, chunk)
                t1 = time.perf_counter()
                times.append(t1 - t0)
            avg = sum(times) / len(times)
            results.append((p, chunk, times[0], times[1], times[2], avg))
            print(f"P={p:2d} chunk={chunk:3d} | run1={times[0]:.4f} run2={times[1]:.4f} run3={times[2]:.4f} | avg={avg:.4f}s")

    with open("task3_2_sweep_matrix.csv", "w") as f:
        f.write("threads,chunk,run1,run2,run3,avg_time_s\n")
        for row in results:
            f.write(",".join(str(x) for x in row) + "\n")
    print("\nSaved task3_2_sweep_matrix.csv")


def task_imbalance(threads, chunk):
    print(f"Task 3.4: Load Imbalance Metric (threads={threads}, chunk={chunk})")
    _ = render_dynamic(50, 50, 100, threads, 5, track_imbalance=True)  # warm-up

    img, iter_counter = render_dynamic(WIDTH, HEIGHT, MAX_ITER, threads, chunk, track_imbalance=True)

    max_work = iter_counter.max()
    min_work = iter_counter.min()
    avg_work = iter_counter.mean()
    imbalance = (max_work - min_work) / avg_work

    print(f"Per-thread total iterations: {list(iter_counter)}")
    print(f"Max={max_work} Min={min_work} Avg={avg_work:.1f}")
    print(f"Imbalance = (Max-Min)/Avg = {imbalance:.4f}")

    with open("task3_4_imbalance.csv", "w") as f:
        f.write("thread_id,total_iterations\n")
        for tid, val in enumerate(iter_counter):
            f.write(f"{tid},{val}\n")
        f.write(f"\n# imbalance_metric,{imbalance}\n")
    print("Saved task3_4_imbalance.csv")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 mandelbrot_lab3.py <static|dynamic|sweep|imbalance> [threads] [chunk]")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "static":
        task_static()
    elif mode == "dynamic":
        threads = int(sys.argv[2])
        chunk = int(sys.argv[3])
        task_dynamic(threads, chunk)
    elif mode == "sweep":
        task_sweep()
    elif mode == "imbalance":
        threads = int(sys.argv[2])
        chunk = int(sys.argv[3])
        task_imbalance(threads, chunk)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)