import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;

public class PiFixed {
    static final long TOTAL_POINTS = 50_000_000L;
    static final int NUM_THREADS = 4;

    // Version A: synchronized
    static long totalHitsSync = 0;
    static synchronized void incrementSync() {
        totalHitsSync++;
    }

    // Version B: AtomicLong
    static AtomicLong totalHitsAtomic = new AtomicLong(0);

    public static void main(String[] args) throws InterruptedException {
        long pointsPerThread = TOTAL_POINTS / NUM_THREADS;

        // --- Single-threaded baseline ---
        long startSingle = System.nanoTime();
        long singleHits = 0;
        Random rndSingle = new Random();
        for (long i = 0; i < TOTAL_POINTS; i++) {
            double x = rndSingle.nextDouble();
            double y = rndSingle.nextDouble();
            if (x * x + y * y <= 1.0) singleHits++;
        }
        long elapsedSingle = System.nanoTime() - startSingle;
        System.out.println("=== Single-threaded baseline ===");
        System.out.println("Pi: " + (4.0 * singleHits / TOTAL_POINTS));
        System.out.println("Time: " + (elapsedSingle / 1_000_000.0) + " ms\n");

        // --- synchronized version ---
        Thread[] threadsSync = new Thread[NUM_THREADS];
        long startSync = System.nanoTime();
        for (int t = 0; t < NUM_THREADS; t++) {
            threadsSync[t] = new Thread(() -> {
                Random rnd = new Random();
                for (long i = 0; i < pointsPerThread; i++) {
                    double x = rnd.nextDouble();
                    double y = rnd.nextDouble();
                    if (x * x + y * y <= 1.0) incrementSync();
                }
            });
            threadsSync[t].start();
        }
        for (Thread th : threadsSync) th.join();
        long elapsedSync = System.nanoTime() - startSync;
        System.out.println("=== synchronized version ===");
        System.out.println("Pi: " + (4.0 * totalHitsSync / TOTAL_POINTS));
        System.out.println("Time: " + (elapsedSync / 1_000_000.0) + " ms\n");

        // --- AtomicLong version ---
        Thread[] threadsAtomic = new Thread[NUM_THREADS];
        long startAtomic = System.nanoTime();
        for (int t = 0; t < NUM_THREADS; t++) {
            threadsAtomic[t] = new Thread(() -> {
                Random rnd = new Random();
                for (long i = 0; i < pointsPerThread; i++) {
                    double x = rnd.nextDouble();
                    double y = rnd.nextDouble();
                    if (x * x + y * y <= 1.0) totalHitsAtomic.incrementAndGet();
                }
            });
            threadsAtomic[t].start();
        }
        for (Thread th : threadsAtomic) th.join();
        long elapsedAtomic = System.nanoTime() - startAtomic;
        System.out.println("=== AtomicLong version ===");
        System.out.println("Pi: " + (4.0 * totalHitsAtomic.get() / TOTAL_POINTS));
        System.out.println("Time: " + (elapsedAtomic / 1_000_000.0) + " ms");
    }
}