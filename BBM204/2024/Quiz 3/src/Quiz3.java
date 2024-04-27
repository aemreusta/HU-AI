import java.util.*;
import java.io.*;

public class Quiz3 {
    static class Station {
        double x, y;

        public Station(double x, double y) {
            this.x = x;
            this.y = y;
        }

        public double distanceTo(Station other) {
            return Math.sqrt(Math.pow(this.x - other.x, 2) + Math.pow(this.y - other.y, 2));
        }
    }

    public static void main(String[] args) throws IOException {
        // Scanner sc = new Scanner(new File(args[0]));
        Scanner sc = new Scanner(new File ("/Users/emre/GitHub/HU-AI/BBM204/2024/Quiz 3/src/sample_io/sample_input_0.txt"));
        int numberOfTestCases = sc.nextInt();

        while (numberOfTestCases-- > 0) {
            int S = sc.nextInt();  // stations with drones
            int P = sc.nextInt();  // total stations

            Station[] stations = new Station[P];
            for (int i = 0; i < P; i++) {
                double x = sc.nextDouble();
                double y = sc.nextDouble();
                stations[i] = new Station(x, y);
            }

            double minimumT = findMinimumT(stations, S);
            System.out.printf("%.2f\n", minimumT);
        }
        sc.close();
    }

    private static double findMinimumT(Station[] stations, int droneCount) {
        List<Double> distances = new ArrayList<>();
        // Calculate distances only between non-drone equipped stations
        for (int i = droneCount; i < stations.length; i++) {
            for (int j = i + 1; j < stations.length; j++) {
                distances.add(stations[i].distanceTo(stations[j]));
            }
        }
        Collections.sort(distances);

        // Implement binary search on the sorted distances
        double left = 0, right = distances.isEmpty() ? 0 : distances.get(distances.size() - 1);
        while (right - left > 1e-6) {
            double mid = (left + right) / 2;
            if (isConnected(stations, mid, droneCount)) {
                right = mid;
            } else {
                left = mid;
            }
        }
        return right;
    }

    private static boolean isConnected(Station[] stations, double T, int droneCount) {
        int n = stations.length;
        boolean[] visited = new boolean[n];
        Stack<Integer> stack = new Stack<>();

        if (droneCount > 0) {
            stack.push(0); // Start from any drone-equipped station
            visited[0] = true;
        }

        while (!stack.isEmpty()) {
            int node = stack.pop();
            for (int neighbor = 0; neighbor < n; neighbor++) {
                if (!visited[neighbor]) {
                    if (node < droneCount || neighbor < droneCount || stations[node].distanceTo(stations[neighbor]) <= T) {
                        stack.push(neighbor);
                        visited[neighbor] = true;
                    }
                }
            }
        }

        // Check connectivity
        for (boolean v : visited) {
            if (!v) return false;
        }
        return true;
    }
}
