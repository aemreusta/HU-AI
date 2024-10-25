import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.StringTokenizer;

public class Main {

    public static void main(String[] args) throws IOException {

        // Read input from standard input
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));

        // Read capacity and number of weights from first line of input
        StringTokenizer tokenizer = new StringTokenizer(reader.readLine());
        int capacity = Integer.parseInt(tokenizer.nextToken());
        int numberOfWeights = Integer.parseInt(tokenizer.nextToken());

        // Read weights from second line of input
        ArrayList<Integer> weights = new ArrayList<Integer>();
        tokenizer = new StringTokenizer(reader.readLine());
        for (int i = 0; i < numberOfWeights; i++) {
            weights.add(Integer.parseInt(tokenizer.nextToken()));
        }

        // Create a two-dimensional boolean array to hold the results of the dynamic
        // programming algorithm
        boolean[][] results = new boolean[capacity + 1][numberOfWeights + 1];

        // Initialize the first column of the array to true
        for (int i = 0; i <= numberOfWeights; i++) {
            results[0][i] = true;
        }

        // Fill in the remaining entries of the array using dynamic programming
        for (int currentCapacity = 1; currentCapacity <= capacity; currentCapacity++) {
            for (int currentWeight = 1; currentWeight <= numberOfWeights; currentWeight++) {
                // If the current weight is greater than the capacity, the value in results is
                // the same as the value in the previous row for the same column
                results[currentCapacity][currentWeight] = results[currentCapacity][currentWeight - 1];
                if (weights.get(currentWeight - 1) <= currentCapacity) {
                    // If the current weight is less than or equal to the capacity, the value in
                    // results is true if either the value in the previous row for the same column
                    // is true or the value in the previous row for the same weight plus the weight
                    // at the current index is true
                    results[currentCapacity][currentWeight] = results[currentCapacity][currentWeight]
                            || results[currentCapacity - weights.get(currentWeight - 1)][currentWeight - 1];
                }
            }
        }

        // Find the maximum weight that can be carried
        int maxWeight = 0;
        for (int currentCapacity = capacity; currentCapacity >= 0; currentCapacity--) {
            if (results[currentCapacity][numberOfWeights]) {
                maxWeight = currentCapacity;
                break;
            }
        }

        // Print the maximum weight that can be carried
        System.out.println(maxWeight);

        // Print the contents of the results array
        for (int currentCapacity = 0; currentCapacity <= capacity; currentCapacity++) {
            for (int currentWeight = 0; currentWeight <= numberOfWeights; currentWeight++) {
                System.out.print(results[currentCapacity][currentWeight] ? "T" : "F");
            }
            System.out.println();
        }

        // Close the BufferedReader
        reader.close();
    }

}
