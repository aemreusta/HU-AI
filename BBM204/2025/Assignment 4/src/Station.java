import java.io.Serializable;
import java.util.Objects;

public class Station implements Serializable {
    static final long serialVersionUID = 55L;

    public Point coordinates;
    public String description; // This should be unique or used carefully as HashMap keys

    public Station(Point coordinates, String description) {
        this.coordinates = coordinates;
        this.description = description;
    }

    public String toString() {
        return this.description;
    }

    // Need equals and hashCode to use Station objects as keys in HashMaps (like in Dijkstra)
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Station station = (Station) o;
        // Assuming description is a unique identifier for a station
        return Objects.equals(description, station.description) &&
               Objects.equals(coordinates, station.coordinates); // Also check coordinates for robustness
    }

    @Override
    public int hashCode() {
        // Use description for hashing, assuming it's unique
        return Objects.hash(description, coordinates);
    }
}