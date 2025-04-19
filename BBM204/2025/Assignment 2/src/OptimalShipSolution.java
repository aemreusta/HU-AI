import java.util.ArrayList;

public class OptimalShipSolution {
    private final ArrayList<Integer> artifactSet;
    private final int solution;

    // Constructor to initialize the artifact set and solution
    public OptimalShipSolution(ArrayList<Integer> artifactSet, int solution) {
        this.artifactSet = artifactSet;
        this.solution = solution;
    }

    // Getter for the artifact set
    public ArrayList<Integer> getArtifactSet() {
        return artifactSet;
    }

    // Getter for the solution (number of spaceships)
    public int getSolution() {
        return solution;
    }

    // Print the solution in the required format
    public void printSolution(OptimalShipSolution solution) {
        System.out.println("Minimum spaceships required: " + solution.getSolution());
        System.out.println("For the artifact set of :" + solution.getArtifactSet());
    }
}
