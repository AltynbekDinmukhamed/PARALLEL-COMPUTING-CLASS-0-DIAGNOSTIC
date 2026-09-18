import java.util.Random;

public class PiReduction {
    static final long TOTAL_POINTS = 100_000_000L;

    static long runWithThreads(int numThreads) throws InterruptedException {
        long pointsPerThread = TOTAL_POINTS / numThreads;
        Thread[] threads = new Thread[numThreads];
        long[] localHits = new long[numThreads]; // each thread writes only its own slot

        for (int t = 0; t < numThreads; t++) {
            final int idx = t;
            threads[t] = new Thread(() -> {
                Random rnd = new Random();
                long count = 0;
                for (long i = 0; i < pointsPerThread; i++) {
                    double x = rnd.nextDouble();
                    double y = rnd.nextDouble();
                    if (x * x + y * y <= 1.0) count++;
                }
                localHits[idx] = count; // written once, after the loop finishes
            });
        }

        for (Thread th : threads) th.start();
        for (Thread th : threads) th.join();

        long total = 0;
        for (long h : localHits) total += h; // single reduction step
        return total;
    }

    public static void main(String[] args) throws InterruptedException {
        int[] threadCounts = {1, 2, 4, 8, 16, 32};
        long baselineTime = 0;

        System.out.println("Threads | Runtime(ms) | Hits | Pi");
        System.out.println("---------------------------------------------");

        for (int t : threadCounts) {
            long start = System.nanoTime();
            long hits = runWithThreads(t);
            long elapsed = (System.nanoTime() - start) / 1_000_000; // ms

            if (t == 1) baselineTime = elapsed;

            double pi = 4.0 * hits / TOTAL_POINTS;
            double speedup = (double) baselineTime / elapsed;
            double efficiency = (speedup / t) * 100;

            System.out.printf("%7d | %11d | %10d | %.6f | Speedup: %.2fx | Efficiency: %.1f%%%n",
                    t, elapsed, hits, pi, speedup, efficiency);
        }
    }
}