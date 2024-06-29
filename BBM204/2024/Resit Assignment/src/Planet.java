import java.io.Serializable;
import java.util.HashSet;
import java.util.List;

public class Planet implements Comparable, Serializable {
    private static final long serialVersionUID = 1L;
    private final String id;
    private final int blackHoleProximity;
    private final List<String> neighbors;

    public Planet(String id, int technologyLevel, List<String> neighbors) {
        this.id = id;
        this.blackHoleProximity = technologyLevel;
        this.neighbors = neighbors;
    }

    public String getId() {
        return id;
    }

    public int getBlackHoleProximity() {
        return blackHoleProximity;
    }

    public List<String> getNeighbors() {
        return neighbors;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Planet planet = (Planet) o;
        return blackHoleProximity == planet.blackHoleProximity &&
                id.equals(planet.id) &&
                new HashSet<>(neighbors).equals(new HashSet<>(((Planet) o).getNeighbors()));
    }

    @Override
    public String toString() {
        return id;
    }

    @Override
    public int compareTo(Object o) {
        Integer ownId = Integer.parseInt(this.getId().substring(1).replace("C",""));
        Integer oId = Integer.parseInt(((Planet) o).getId().substring(1).replace("C",""));
        return ownId.compareTo(oId);
    }
}

