import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class SolarSystem implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private final List<Planet> planets = new ArrayList<>();

    public boolean hasPlanet(String planetId) {
        return planets.stream().map(Planet::getId).collect(Collectors.toList()).contains(planetId);
    }

    public void addPlanet(Planet planet) {
        planets.add(planet);
    }

    public List<Planet> getPlanets() {
        return planets;
    }
}
