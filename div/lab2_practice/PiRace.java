import java.util.Random;

public class PiRace {
    static long totalHits = 0;
    static final long TOTAL_POINTS = 50_000_000L;
    static final int NUM_THREADS = 4;

    public static void main(String[] args) throws InterruptedException {
        long pointsPerThread = TOTAL_POINTS / NUM_THREADS;
        Thread[] threads = new Thread[NUM_THREADS];

        long start = System.nanoTime();

        for (int t = 0; t < NUM_THREADS; t++) {
            threads[t] = new Thread(() -> {
                Random rnd = new Random();
                for (long i = 0; i < pointsPerThread; i++) {
                    double x = rnd.nextDouble();
                    double y = rnd.nextDouble();
                    if (x * x + y * y <= 1.0) {
                        totalHits++;  // <-- NOT thread-safe, this is the bug
                    }
                }
            });
            threads[t].start();
        }

        for (Thread th : threads) th.join();

        long elapsed = System.nanoTime() - start;
        double pi = 4.0 * totalHits / TOTAL_POINTS;

        System.out.println("Total Hits: " + totalHits);
        System.out.println("Estimated Pi: " + pi);
        System.out.println("Time: " + (elapsed / 1_000_000.0) + " ms");
    }
}