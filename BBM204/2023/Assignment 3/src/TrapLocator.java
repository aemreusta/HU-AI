import java.util.*;

public class TrapLocator {
    public List<Colony> colonies;

    public TrapLocator(List<Colony> colonies) {
        this.colonies = colonies;
    }

    public List<List<Integer>> revealTraps() {

        // Trap positions for each colony, should contain an empty array if the colony is safe.
        // I.e.:
        // 0 -> [2, 15, 16, 31]
        // 1 -> [4, 13, 22]
        // 3 -> []
        // ...
        List<List<Integer>> traps = new ArrayList<>();

        // Identify the time traps and save them into the traps variable and then return it.
        // TODO: Your code here

        return traps;
    }

    public void printTraps(List<List<Integer>> traps) {
        // For each colony, if you have encountered a time trap, then print the cities that create the trap.
        // If you have not encountered a time trap in this colony, then print "Safe".
        // Print the your findings conforming to the given output format.
        // TODO: Your code here
    }

}
