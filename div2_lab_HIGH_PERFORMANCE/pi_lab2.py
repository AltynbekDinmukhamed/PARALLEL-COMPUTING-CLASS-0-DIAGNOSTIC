import sys
import time
import math
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from numba import njit, prange

TRUE_PI = math.pi


# ---------------- Serial baseline ----------------
@njit
def calc_pi_serial(num_steps: int) -> float:
    step = 1.0 / num_steps
    total = 0.0
    for i in range(num_steps):
        x = (i + 0.5) * step
        total += 4.0 / (1.0 + x * x)
    return total * step


# ---------------- Variant C: parallel reduction ----------------
@njit(parallel=True)
def calc_pi_reduction(num_steps: int) -> float:
    step = 1.0 / num_steps
    total = 0.0
    for i in prange(num_steps):
        x = (i + 0.5) * step
        total += 4.0 / (1.0 + x * x)
    return total * step


# ---------------- Variant A: naive race (real race via nogil) ----------------
@njit(nogil=True)
def _race_chunk(start, end, step, shared_sum, idx):
    # unprotected read-modify-write on shared_sum[idx] (idx is always 0 — single shared slot)
    for i in range(start, end):
        x = (i + 0.5) * step
        term = 4.0 / (1.0 + x * x)
        shared_sum[idx] += term  # NOT atomic: real race across threads


def naive_race(num_steps: int, threads: int) -> float:
    step = 1.0 / num_steps
    shared_sum = np.zeros(1, dtype=np.float64)
    chunk = num_steps // threads

    with ThreadPoolExecutor(max_workers=threads) as ex:
        futures = []
        for t in range(threads):
            start = t * chunk
            end = num_steps if t == threads - 1 else start + chunk
            futures.append(ex.submit(_race_chunk, start, end, step, shared_sum, 0))
        for f in futures:
            f.result()

    return shared_sum[0] * step


# ---------------- Variant B: critical section (pure Python, locked) ----------------
def critical_section(num_steps: int, threads: int) -> float:
    step = 1.0 / num_steps
    lock = threading.Lock()
    shared_sum = [0.0]
    chunk = num_steps // threads

    def worker(start, end):
        for i in range(start, end):
            x = (i + 0.5) * step
            term = 4.0 / (1.0 + x * x)
            with lock:
                shared_sum[0] += term

    threads_list = []
    for t in range(threads):
        start = t * chunk
        end = num_steps if t == threads - 1 else start + chunk
        th = threading.Thread(target=worker, args=(start, end))
        threads_list.append(th)
        th.start()
    for th in threads_list:
        th.join()

    return shared_sum[0] * step


# ---------------- Task runners ----------------
def task_race():
    print("Task 2.1: Race Condition Quantification (N=100,000,000)")
    print("Using numba(nogil=True) threads -> a REAL hardware race, unlike plain Python threading.\n")
    N = 100_000_000
    # warm-up JIT
    _race_chunk(0, 10, 1.0 / N, np.zeros(1), 0)

    rows = []
    for p in [1, 2, 4, 8]:
        pi_val = naive_race(N, p)
        error = abs(pi_val - TRUE_PI)
        rows.append((p, pi_val, error))
        print(f"P={p} | Pi = {pi_val:.12f} | Error = {error:.6e}")

    with open("task2_1_race_errors.csv", "w") as f:
        f.write("threads,pi_value,abs_error\n")
        for p, pi_val, err in rows:
            f.write(f"{p},{pi_val},{err}\n")
    print("\nSaved task2_1_race_errors.csv")


def task_critical():
    print("Task 2.2: Critical Section Overhead (N=1,000,000)")
    N = 1_000_000
    # warm-up numba serial baseline
    _ = calc_pi_serial(1000)

    t0 = time.perf_counter()
    pi_serial = calc_pi_serial(N)
    t1 = time.perf_counter()
    time_serial = t1 - t0

    t2 = time.perf_counter()
    pi_crit = critical_section(N, 4)
    t3 = time.perf_counter()
    time_crit = t3 - t2

    overhead_pct = (time_crit - time_serial) / time_serial * 100

    print(f"Serial (numba, 1 thread):     Pi = {pi_serial:.10f} | Time = {time_serial:.4f}s")
    print(f"Critical section (4 threads): Pi = {pi_crit:.10f} | Time = {time_crit:.4f}s")
    print(f"Lock contention overhead: {overhead_pct:.1f}% slower than serial baseline")

    with open("task2_2_critical_section.csv", "w") as f:
        f.write("variant,time_s,pi_value\n")
        f.write(f"serial,{time_serial},{pi_serial}\n")
        f.write(f"critical_4threads,{time_crit},{pi_crit}\n")
    print("Saved task2_2_critical_section.csv")


def task_reduction():
    print("Task 2.3 + 2.4: Strong Scaling Benchmark (N=100,000,000), P in {1,2,4,8,16}")
    N = 100_000_000
    # warm-up JIT
    _ = calc_pi_reduction(1000)

    from numba import set_num_threads
    results = []
    t1_time = None
    for p in [1, 2, 4, 8, 16]:
        set_num_threads(min(p, 8))  # cap at physical/logical core count on this machine
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            pi_val = calc_pi_reduction(N)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        avg_t = sum(times) / len(times)
        if p == 1:
            t1_time = avg_t
        speedup = t1_time / avg_t
        efficiency = speedup / p
        results.append((p, avg_t, speedup, efficiency, pi_val))
        print(f"P={p:2d} | avg_time={avg_t:.4f}s | S(P)={speedup:.3f} | E(P)={efficiency:.3f} | Pi={pi_val:.8f}")

    with open("task2_3_reduction_scaling.csv", "w") as f:
        f.write("P,avg_time_s,speedup,efficiency,pi_value\n")
        for row in results:
            f.write(",".join(str(x) for x in row) + "\n")
    print("\nSaved task2_3_reduction_scaling.csv")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 pi_lab2.py <race|critical|reduction>")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "race":
        task_race()
    elif mode == "critical":
        task_critical()
    elif mode == "reduction":
        task_reduction()
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)