import java.util.*;
import java.io.*;

public class Quiz3 {
    public static void main(String[] args) throws IOException {

        // String filename = args[0];
        String filename = "/Users/emre/GitHub/HU-AI/BBM204/2024/Quiz 3/src/sample_io/sample_input_1.txt";
        try (Scanner scanner = new Scanner(new File(filename))) {
            int testCases = scanner.nextInt();
            for (int t = 0; t < testCases; t++) {
                int S = scanner.nextInt(); // Stations with drones
                int P = scanner.nextInt(); // Total stations

                int[][] stations = new int[P][2];
                for (int i = 0; i < P; i++) {
                    stations[i][0] = scanner.nextInt(); // x coordinate
                    stations[i][1] = scanner.nextInt(); // y coordinate
                }

                // List to store the minimum distances for each station without a drone
                List<Double> minDistances = new ArrayList<>();

                // Determine which stations are equipped with drones
                boolean[] hasDrone = new boolean[P];
                Arrays.fill(hasDrone, false);
                for (int i = 0; i < S; i++) {
                    hasDrone[i] = true;
                }

                // Calculate distances only for pairs that do not both have drones
                for (int i = 0; i < P; i++) {
                    double minDistance = Double.MAX_VALUE;
                    for (int j = 0; j < P; j++) {
                        if (i != j && !(hasDrone[i] && hasDrone[j])) {
                            double dist = Math.sqrt(Math.pow(stations[i][0] - stations[j][0], 2) 
                                                  + Math.pow(stations[i][1] - stations[j][1], 2));
                            if (dist < minDistance) {
                                minDistance = dist;
                            }
                        }
                    }
                    // Only add to list if station does not have a drone
                    if (!hasDrone[i]) {
                        minDistances.add(minDistance);
                    }
                }

                // Sort the list of minimum distances
                Collections.sort(minDistances);

                // The result is the largest of the remaining distances
                double result = minDistances.isEmpty() ? 0 : minDistances.get(minDistances.size() - 1);

                System.out.printf(Locale.US, "%.2f\n", result);
            }
        } catch (FileNotFoundException e) {
            System.out.println("File not found: " + filename);
        }
    }
}
