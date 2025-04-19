import java.util.ArrayList;
import java.util.Collections;

public class MinShipsGP {
    private final ArrayList<Integer> artifactsFound;

    public MinShipsGP(ArrayList<Integer> artifactsFound) {
        this.artifactsFound = artifactsFound;
    }

    public ArrayList<Integer> getArtifactsFound() {
        return artifactsFound;
    }

    public OptimalShipSolution optimalArtifactCarryingAlgorithm() {
        ArrayList<Integer> sortedArtifacts = new ArrayList<>(artifactsFound);
        Collections.sort(sortedArtifacts, Collections.reverseOrder());

        ArrayList<Integer> spaceships = new ArrayList<>();

        for (int weight : sortedArtifacts) {
            boolean placed = false;
            for (int i = 0; i < spaceships.size(); i++) {
                if (spaceships.get(i) + weight <= 100) {
                    spaceships.set(i, spaceships.get(i) + weight);
                    placed = true;
                    break;
                }
            }
            if (!placed) {
                spaceships.add(weight);
            }
        }

        return new OptimalShipSolution(artifactsFound, spaceships.size());
    }
}