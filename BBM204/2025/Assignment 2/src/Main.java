import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) throws IOException {

        // Uncomment the following lines to read file paths from command line arguments
        File safesFile = new File(args[0]);
        File artifactsFile = new File(args[1]);

        // Tests
        // File safesFile = new File("/Users/emre/GitHub/HU-AI/BBM204/2025/Assignment 2/src/io/SafesDiscovered.dat");
        // File artifactsFile = new File("/Users/emre/GitHub/HU-AI/BBM204/2025/Assignment 2/src/io/ArtifactsFound.dat");

        /** Safe-lock Opening Algorithm **/
        System.out.println("##Initiate Operation Safe-lock##");
        ArrayList<ArrayList<Integer>> safesDiscovered = new ArrayList<>();
        try (Scanner scanner = new Scanner(safesFile)) {
            int numSafes = Integer.parseInt(scanner.nextLine().trim());
            for (int i = 0; i < numSafes; i++) {
                String[] parts = scanner.nextLine().trim().split(",");
                ArrayList<Integer> safe = new ArrayList<>();
                safe.add(Integer.parseInt(parts[0].trim()));
                safe.add(Integer.parseInt(parts[1].trim()));
                safesDiscovered.add(safe);
            }
        } catch (FileNotFoundException e) {
            System.err.println("Safes file not found: " + e.getMessage());
            return;
        }

        MaxScrollsDP maxScrollsDP = new MaxScrollsDP(safesDiscovered);
        OptimalScrollSolution optimalScrollSolution = maxScrollsDP.optimalSafeOpeningAlgorithm();
        optimalScrollSolution.printSolution(optimalScrollSolution);
        System.out.println("##Operation Safe-lock Completed##");

        /** Artifact Carrying Algorithm **/
        System.out.println("##Initiate Operation Artifact##");
        ArrayList<Integer> artifactsFound = new ArrayList<>();
        try (Scanner scanner = new Scanner(artifactsFile)) {
            if (scanner.hasNextLine()) {
                String line = scanner.nextLine().trim();
                String[] parts = line.split(",");
                for (String part : parts) {
                    artifactsFound.add(Integer.parseInt(part.trim()));
                }
            }
        } catch (FileNotFoundException e) {
            System.err.println("Artifacts file not found: " + e.getMessage());
            return;
        }

        MinShipsGP minShipsGP = new MinShipsGP(artifactsFound);
        OptimalShipSolution optimalShipSolution = minShipsGP.optimalArtifactCarryingAlgorithm();
        optimalShipSolution.printSolution(optimalShipSolution);
        System.out.println("##Operation Artifact Completed##");
    }
}