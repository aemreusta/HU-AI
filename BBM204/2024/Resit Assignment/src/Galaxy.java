import java.io.Serializable;
import java.util.*;

public class Galaxy implements Serializable  {

    private transient static final long serialVersionUID = 1L;
    private final List<Planet> planets;
    private List<SolarSystem> solarSystems;

    public Galaxy(List<Planet> planets) {
        this.planets = planets;
    }

     /**
     * Using the galaxy's list of Planet objects, explores all the solar systems in the galaxy with respect to the rules regarding black holes and duplicators.
     * Saves the result to the solarSystems instance variable and returns a shallow copy of it.
     *
     * @return List of SolarSystem objects.
     */
    public List<SolarSystem> exploreSolarSystems() {
        solarSystems = new ArrayList<>();
        Set<String> visitedPlanets = new HashSet<>();
        // list of the planets has too much black hole proximity
        List<Planet> highBlackHoleProximityPlanets = new ArrayList<>();

        for (Planet planet : planets) {
            String planetId = planet.getId();

            // Skip planet if already visited
            if (visitedPlanets.contains(planetId)) {
                continue;
            }

            // if planet has high black hole proximity, add it to the list and its duplicator planet if it has and neighbors
            if (planet.getBlackHoleProximity() > 90) {
                highBlackHoleProximityPlanets.add(planet);

                if (hasDuplicatorPlanet(planet)) {
                    Planet duplicatorPlanet = findPlanetById(planetId + "C");
                    if (duplicatorPlanet != null) {
                        highBlackHoleProximityPlanets.add(duplicatorPlanet);
                    }
                }

                for (String neighborId : planet.getNeighbors()) {
                    Planet neighborPlanet = findPlanetById(neighborId);
                    if (neighborPlanet != null) {
                        highBlackHoleProximityPlanets.add(neighborPlanet);
                    }
                }
            }

            // Create a new solar system starting from this planet
            SolarSystem solarSystem = new SolarSystem();
            exploreSolarSystemDFS(planet, visitedPlanets, solarSystem, highBlackHoleProximityPlanets);

            // Add solar system to the list if it's not empty and does not contain any planet higher than 90 BlackHoleProximity and contains at two planets
            if (!solarSystem.getPlanets().isEmpty() && checkSolarSystemValidity(solarSystem, highBlackHoleProximityPlanets) ){
                solarSystems.add(solarSystem);
            }
        }

        return solarSystems;
    }

    // check if planet has a duplicator planet in planets
    private boolean hasDuplicatorPlanet(Planet planet) {
        for (Planet p : planets) {
            if (p.getId().equals(planet.getId() + "C")) {
                return true;
            }
        }
        return false;
    }

    private void exploreSolarSystemDFS(Planet currentPlanet, Set<String> visitedPlanets, SolarSystem solarSystem, List<Planet> highBlackHoleProximityPlanets) {
        String currentPlanetId = currentPlanet.getId();

        // Add current planet to the solar system
        solarSystem.addPlanet(currentPlanet);

        // Mark current planet as visited
        visitedPlanets.add(currentPlanetId);

        // if current planet has a duplicator planet, add it to the solar system
        if (hasDuplicatorPlanet(currentPlanet)) {
            Planet duplicatorPlanet = findPlanetById(currentPlanetId + "C");
            if (duplicatorPlanet != null) {
                solarSystem.addPlanet(duplicatorPlanet);
                visitedPlanets.add(duplicatorPlanet.getId());
            }
        }

        // Explore neighbors
        for (String neighborId : currentPlanet.getNeighbors()) {
            if (!visitedPlanets.contains(neighborId)) {
                // Find the neighbor planet object
                Planet neighborPlanet = findPlanetById(neighborId);
                if (neighborPlanet != null) {
                        // Recursive DFS for the neighbor planet
                    exploreSolarSystemDFS(neighborPlanet, visitedPlanets, solarSystem, highBlackHoleProximityPlanets);
                }
            }
        }
    }

    private Planet findPlanetById(String id) {
        for (Planet planet : planets) {
            if (planet.getId().equals(id)) {
                return planet;
            }
        }
        return null;
    }

    private boolean checkSolarSystemValidity(SolarSystem solarSystem, List<Planet> highBlackHoleProximityPlanets) {
        for (Planet planet : solarSystem.getPlanets()) {
            if (planet.getBlackHoleProximity() > 90) {
                return false; // Invalid solar system if any planet has high BlackHoleProximity
            }

            if (highBlackHoleProximityPlanets.contains(planet)) {
                return false; // Invalid solar system if any planet is in the highBlackHoleProximityPlanets list
            }

            if (hasDuplicatorPlanet(planet) && highBlackHoleProximityPlanets.contains(findPlanetById(planet.getId() + "C"))) {
                return false; // Invalid solar system if any duplicator planet is in the highBlackHoleProximityPlanets list
            }

            for (String neighborId : planet.getNeighbors()) {
                Planet neighborPlanet = findPlanetById(neighborId);
                if (neighborPlanet != null && highBlackHoleProximityPlanets.contains(neighborPlanet)) {
                    return false; // Invalid solar system if any neighbor planet is in the highBlackHoleProximityPlanets list
                }
            }
        }
        return true;
    }

    public List<SolarSystem> getSolarSystems() {
        return solarSystems;
    }

    // FOR TESTING
    public List<Planet> getPlanets() {
        return planets;
    }

    // FOR TESTING
    public int getSolarSystemIndexByPlanetID(String planetId) {
        for (int i = 0; i < solarSystems.size(); i++) {
            SolarSystem solarSystem = solarSystems.get(i);
            if (solarSystem.hasPlanet(planetId)) {
                return i;
            }
        }
        return -1;
    }

    public void printSolarSystems(List<SolarSystem> solarSystems) {
        System.out.printf("%d solar systems have been discovered.%n", solarSystems.size());
        for (int i = 0; i < solarSystems.size(); i++) {
            SolarSystem solarSystem = solarSystems.get(i);
            List<Planet> planets = new ArrayList<>(solarSystem.getPlanets());
            Collections.sort(planets);
            System.out.printf("Planets in Solar System %d: %s", i + 1, planets);
            System.out.println();
        }
    }
}
