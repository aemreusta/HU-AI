import java.util.*;
import java.io.*;

public class Quiz2 {
    public static void main(String[] args) throws IOException {
        // Replace hardcoded file path with command line argument for flexibility
        String filePath = args[0];

        // Initialize scanner to read from file
        Scanner in = new Scanner(new File(filePath));
        
        // Read spacecraft's capacity (M) and number of resources (n)
        int M = in.nextInt(); 
        int n = in.nextInt();
        
        // Initialize an array to hold the masses of resources
        int[] masses = new int[n];
        for (int i = 0; i < n; i++) {
            masses[i] = in.nextInt(); // Populate the array with resource masses
        }
        in.close(); // Close the scanner

        // Initialize dynamic programming matrix, L, to track loadable combinations
        boolean[][] L = new boolean[M + 1][n + 1];
        for (int i = 0; i <= n; i++) {
            L[0][i] = true; // Base case: 0 mass is always loadable
        }

        // Fill the DP matrix to find combinations of resources that can be optimally loaded
        for (int i = 1; i <= n; i++) {
            for (int m = 1; m <= M; m++) {
                // Inherit loadability from previous resource
                L[m][i] = L[m][i - 1];
                
                // Check if current resource can be added without exceeding capacity
                if (masses[i - 1] <= m) {
                    // Update loadability by including current resource if it's beneficial
                    L[m][i] = L[m][i] || L[m - masses[i - 1]][i - 1];
                }
            }
        }

        // Determine the maximum mass that can be loaded without exceeding capacity
        int maxMass = 0;
        for (int m = M; m >= 0; m--) {
            if (L[m][n]) {
                maxMass = m; // Find the highest mass that can be loaded
                break; // Exit loop once found
            }
        }

        // Output the maximum loadable mass
        System.out.println(maxMass);
        
        // Print the DP matrix to STDOUT, representing loadable mass combinations
        for (int m = 0; m <= M; m++) {
            for (int i = 0; i <= n; i++) {
                System.out.print(L[m][i] ? 1 : 0); // Use ternary operator for concise conditional printing
            }
            System.out.println(); // New line after each mass level for clarity
        }
    }
}
