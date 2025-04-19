import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) throws IOException {

        // File safesFile = new File(args[0]);
        // File artifactsFile = new File(args[1]);

        // Tests
        File safesFile = new File("/Users/emre/GitHub/HU-AI/BBM204/2025/Assignment 2/src/io/SafesDiscovered.dat");
        File artifactsFile = new File("/Users/emre/GitHub/HU-AI/BBM204/2025/Assignment 2/src/io/ArtifactsFound.dat");

        /** Safe-lock Opening Algorithm Below **/

        System.out.println("##Initiate Operation Safe-lock##");

        // Reading the safes data from file
        ArrayList<ArrayList<Integer>> safesDiscovered = new ArrayList<>();
        try {
            Scanner scanner = new Scanner(safesFile); // The first argument should be the file for safes
            int numSafes = Integer.parseInt(scanner.nextLine().trim()); // Read the number of safes

            for (int i = 0; i < numSafes; i++) {
                String line = scanner.nextLine().trim();
                String[] parts = line.split(",");
                ArrayList<Integer> safe = new ArrayList<>();
                safe.add(Integer.parseInt(parts[0].trim())); // Complexity value
                safe.add(Integer.parseInt(parts[1].trim())); // Scroll count
                safesDiscovered.add(safe);
            }
            scanner.close();
        } catch (FileNotFoundException e) {
            System.err.println("Safes file not found: " + e.getMessage());
            return;
        }

        // Instantiate MaxScrollsDP and solve
        MaxScrollsDP maxScrollsDP = new MaxScrollsDP(safesDiscovered);
        OptimalScrollSolution optimalScrollSolution = maxScrollsDP.optimalSafeOpeningAlgorithm();
        optimalScrollSolution.printSolution(optimalScrollSolution);

        System.out.println("##Operation Safe-lock Completed##");

        /** Operation Artifact Algorithm Below **/

        System.out.println("##Initiate Operation Artifact##");

        // Reading the artifacts data from file
        ArrayList<Integer> artifactsFound = new ArrayList<>();
        try {
            Scanner scanner = new Scanner(artifactsFile); // The second argument should be the file for artifacts
            while (scanner.hasNextInt()) {
                artifactsFound.add(scanner.nextInt());
            }
            scanner.close();
        } catch (FileNotFoundException e) {
            System.err.println("Artifacts file not found: " + e.getMessage());
            return;
        }

        // Instantiate MinShipsGP and solve
        MinShipsGP minShipsGP = new MinShipsGP(artifactsFound);
        OptimalShipSolution optimalShipSolution = minShipsGP.optimalArtifactCarryingAlgorithm();
        optimalShipSolution.printSolution(optimalShipSolution);

        System.out.println("##Operation Artifact Completed##");

    }
}
