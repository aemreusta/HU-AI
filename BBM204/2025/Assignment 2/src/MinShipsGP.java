import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

public class MinShipsGP {
    private final ArrayList<Integer> artifactsFound = new ArrayList<>();
    // Weight of artifacts as list will be provided in the input file, and the list
    // should be populated using this format.
    // [3,2,3,4,5,4]

    public ArrayList<Integer> getArtifactsFound() {
        return artifactsFound;
    }

    MinShipsGP(ArrayList<Integer> artifactsFound) {
        this.artifactsFound.addAll(artifactsFound);
    }

    public OptimalShipSolution optimalArtifactCarryingAlgorithm() throws FileNotFoundException {
        // Implement your greedy programming algorithm using the equation 2
        // provided in the assignment file.
    }
}
