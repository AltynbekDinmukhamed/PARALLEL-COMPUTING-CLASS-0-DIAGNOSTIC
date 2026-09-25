import sys
import time
import threading
import math
from concurrent.futures import ThreadPoolExecutor


def worker_task(thread_id: int, team_size: int):
    native_tid = threading.get_native_id()
    role = "Master" if thread_id == 0 else "Worker"
    time.sleep(0.001 * (thread_id % 3))
    print(f"[{role}] Logical Rank: {thread_id} of {team_size} | Native OS TID: {native_tid}")


def run_team(num_threads: int):
    print(f"--- Forking a team of {num_threads} threads ---")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker_task, tid, num_threads) for tid in range(num_threads)]
        for f in futures:
            f.result()
    print("--- Joined thread team. Execution returned to serial master ---\n")


# ---------- Task 1.1: Verification of Non-Determinism ----------
def task_determinism():
    """Run the team 10 times in a row, unmodified. Save stdout to a file too."""
    log_lines = []
    import io
    for run in range(1, 11):
        print(f"\n========== RUN {run} ==========")
        log_lines.append(f"\n========== RUN {run} ==========")
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            run_team(4)
        finally:
            sys.stdout = old_stdout
        text = buf.getvalue()
        print(text, end="")
        log_lines.append(text)

    with open("task1_1_determinism_log.txt", "w") as f:
        f.write("\n".join(log_lines))
    print("\nSaved raw stdout of all 10 runs to task1_1_determinism_log.txt")


# ---------- Task 1.2: Thread Oversubscription Sweep ----------
def task_oversubscription():
    """Measure wall-clock time to instantiate/join a team, for P in {1,2,4,8,16,32,64}."""
    results = []
    for p in [1, 2, 4, 8, 16, 32, 64]:
        # warm-up
        run_team_silent(p)
        # 3 timed runs
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            run_team_silent(p)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        avg = sum(times) / len(times)
        results.append((p, times[0], times[1], times[2], avg))
        print(f"P={p:3d} | run1={times[0]:.6f}s run2={times[1]:.6f}s run3={times[2]:.6f}s | avg={avg:.6f}s")

    with open("task1_2_oversubscription.csv", "w") as f:
        f.write("P,run1,run2,run3,avg_time_s\n")
        for row in results:
            f.write(",".join(str(x) for x in row) + "\n")
    print("Saved timing table to task1_2_oversubscription.csv")


def run_team_silent(num_threads: int):
    """Same as run_team but without prints, for clean timing."""
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(lambda tid=t: threading.get_native_id()) for t in range(num_threads)]
        for f in futures:
            f.result()


# ---------- Task 1.3: CPU Core Saturation Analysis ----------
def cpu_heavy_work():
    """10,000,000 floating-point square roots — artificial workload per thread."""
    total = 0.0
    for i in range(10_000_000):
        total += math.sqrt(i)
    return total


def task_saturation():
    print("Starting CPU saturation test with 4 threads.")
    print("While this runs, open Activity Monitor and watch CPU usage per core.")
    print("(This will take a while — CPU-bound work on each thread.)\n")

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(cpu_heavy_work) for _ in range(4)]
        for f in futures:
            f.result()
    t1 = time.perf_counter()

    print(f"Completed 4 parallel CPU-heavy tasks in {t1 - t0:.4f} seconds.")
    print("NOTE: Because of Python's GIL, threading does NOT give true CPU")
    print("parallelism for pure-Python CPU-bound code. Compare this time to")
    print("running cpu_heavy_work() 4 times sequentially (see below) to see")
    print("the GIL's effect firsthand.")

    t2 = time.perf_counter()
    for _ in range(4):
        cpu_heavy_work()
    t3 = time.perf_counter()
    print(f"Sequential baseline (4x, single thread): {t3 - t2:.4f} seconds")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 fork_join_lab1.py <determinism|oversubscription|saturation>")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "determinism":
        task_determinism()
    elif mode == "oversubscription":
        task_oversubscription()
    elif mode == "saturation":
        task_saturation()
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)