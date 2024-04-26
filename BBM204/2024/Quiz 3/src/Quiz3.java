import java.util.*;
import java.io.*;

public class Quiz3 {
    public static void main(String[] args) throws IOException {
        // Scanner scanner = new Scanner(new File(args[0])); // Adjust the path if running locally
        Scanner scanner = new Scanner(new File("/Users/emre/GitHub/HU-AI/BBM204/2024/Quiz 3/src/sample_io/sample_input_0.txt"));
        int numberOfTestCases = scanner.nextInt(); // First number is the number of test cases

        for (int t = 0; t < numberOfTestCases; t++) {
            int S = scanner.nextInt(); // Number of stations with drones
            int P = scanner.nextInt(); // Total number of stations
            int[][] stations = new int[P][2];

            for (int i = 0; i < P; i++) {
                stations[i][0] = scanner.nextInt(); // X coordinate
                stations[i][1] = scanner.nextInt(); // Y coordinate
            }

            double result = calculateMinimumT(stations, S, P);
            System.out.printf("%.2f\n", result);
        }
        scanner.close();
    }

    private static double calculateMinimumT(int[][] stations, int S, int P) {
        // Priority queue to manage the edges for Kruskal's algorithm
        PriorityQueue<Edge> edgeQueue = new PriorityQueue<>(Comparator.comparingDouble(e -> e.weight));
        // Parent array for union-find
        int[] parent = new int[P];
        for (int i = 0; i < P; i++) {
            parent[i] = i;
        }

        // Add edges between all pairs of stations
        for (int i = 0; i < P; i++) {
            for (int j = i + 1; j < P; j++) {
                double weight = calculateDistance(stations[i], stations[j]);
                if (i < S && j < S) { // Both stations have drones
                    weight = 0; // Zero distance for drone-to-drone connections
                }
                edgeQueue.add(new Edge(i, j, weight));
            }
        }

        // Kruskal's algorithm to find the MST
        double maxEdge = 0;
        int edgesUsed = 0;
        while (!edgeQueue.isEmpty() && edgesUsed < P - 1) {
            Edge edge = edgeQueue.poll();
            if (union(edge.u, edge.v, parent)) {
                maxEdge = Math.max(maxEdge, edge.weight);
                edgesUsed++;
            }
        }

        return maxEdge;
    }

    private static double calculateDistance(int[] a, int[] b) {
        return Math.sqrt(Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2));
    }

    private static int find(int i, int[] parent) {
        if (parent[i] != i) {
            parent[i] = find(parent[i], parent);
        }
        return parent[i];
    }

    private static boolean union(int u, int v, int[] parent) {
        int rootU = find(u, parent);
        int rootV = find(v, parent);
        if (rootU != rootV) {
            parent[rootV] = rootU;
            return true;
        }
        return false;
    }

    static class Edge {
        int u, v;
        double weight;

        Edge(int u, int v, double weight) {
            this.u = u;
            this.v = v;
            this.weight = weight;
        }
    }
}
