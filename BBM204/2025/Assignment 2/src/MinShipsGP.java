import java.util.ArrayList;
import java.util.Collections;

public class MinShipsGP {
    private final ArrayList<Integer> artifactsFound;

    public MinShipsGP(ArrayList<Integer> artifactsFound) {
        this.artifactsFound = artifactsFound;
    }

    public OptimalShipSolution optimalArtifactCarryingAlgorithm() {
        ArrayList<Integer> sorted = new ArrayList<>(artifactsFound);
        Collections.sort(sorted, Collections.reverseOrder());
        ArrayList<Integer> ships = new ArrayList<>();

        for (int weight : sorted) {
            boolean placed = false;
            for (int i = 0; i < ships.size(); i++) {
                if (ships.get(i) + weight <= 100) {
                    ships.set(i, ships.get(i) + weight);
                    placed = true;
                    break;
                }
            }
            if (!placed) {
                ships.add(weight);
            }
        }
        return new OptimalShipSolution(artifactsFound, ships.size());
    }
}