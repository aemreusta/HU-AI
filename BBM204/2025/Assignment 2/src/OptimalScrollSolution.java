import java.util.ArrayList;

public class OptimalScrollSolution {
    private final ArrayList<ArrayList<Integer>> safesDiscovered;
    private final int solution;

    // Constructor to initialize the safes discovered and solution
    public OptimalScrollSolution(ArrayList<ArrayList<Integer>> safesDiscovered, int solution) {
        this.safesDiscovered = safesDiscovered;
        this.solution = solution;
    }

    // Getter for the safes discovered list
    public ArrayList<ArrayList<Integer>> getSafesDiscovered() {
        return safesDiscovered;
    }

    // Getter for the solution (maximum number of scrolls)
    public int getSolution() {
        return solution;
    }

    // Print the solution in the required format
    public void printSolution(OptimalScrollSolution solution) {
        System.out.println("Maximum scrolls acquired: " + solution.getSolution());
        System.out.println("For the safe set of :" + solution.getSafesDiscovered());
    }
}
