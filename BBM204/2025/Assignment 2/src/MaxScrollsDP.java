import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class MaxScrollsDP {
    private ArrayList<ArrayList<Integer>> safesDiscovered = new ArrayList<>();
    // Input format will be the same as following:
    // Number of safes
    // [Complexity,Scroll] Pair
    // [Complexity,Scroll] Pair
    // .
    // .
    // .
    // [Complexity,Scroll] Pair
    // See example provided below:
    // 3
    // [5,10]
    // [10,10]
    // [5,20]

    public MaxScrollsDP(ArrayList<ArrayList<Integer>> safesDiscovered) {
        this.safesDiscovered = safesDiscovered;
    }

    public ArrayList<ArrayList<Integer>> getSafesDiscovered() {
        return safesDiscovered;
    }

    public OptimalScrollSolution optimalSafeOpeningAlgorithm() throws FileNotFoundException {
        // Implement your dynamic programming algorithm using the equation 1
        // provided in the assignment file.
    }
}
