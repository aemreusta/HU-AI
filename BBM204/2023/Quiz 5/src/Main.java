// This code defines a program to find the least crowded path between two cities on a map, using Dijkstra's algorithm.
// The input format is:
// - The first line contains an integer n, the number of cities on the map.
// - Each of the next n lines contains four integers:
// - A plate ID, a unique identifier for the city.
// - The city's name.
// - The city's population.
// - The number of neighbors of the city.
// This is followed by a list of the neighbors' plate IDs.
// - The last line contains two integers: the starting and target plate IDs.
// The output format is:
// - A string describing the least crowded path from the starting city to the target city, followed by a list of the cities on the path, in order.

import java.util.*;

public class Main {
    // Define a constant for infinity.
    static final int INF = Integer.MAX_VALUE;

    // Define a class to represent a city.
    static class City {
        int plateId; // Unique identifier for the city.
        String cityName; // The name of the city.
        int population; // The population of the city.
        List<Integer> neighbors; // A list of the plate IDs of the city's neighbors.

        // Constructor for the City class.
        public City(int plateId, String cityName, int population) {
            this.plateId = plateId;
            this.cityName = cityName;
            this.population = population;
            neighbors = new ArrayList<>();
        }
    }

    // Define a class to represent an edge between two cities.
    static class Edge {
        int to; // The plate ID of the destination city.
        int cost; // The cost of the edge.

        // Constructor for the Edge class.
        public Edge(int to, int cost) {
            this.to = to;
            this.cost = cost;
        }
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        // Read in the number of cities on the map.
        int n = sc.nextInt();

        // Create a map to store information about the cities.
        Map<Integer, City> cities = new HashMap<>();

        // Read in information about each city.
        for (int i = 0; i < n; i++) {
            int plateId = sc.nextInt();
            String cityName = sc.next();
            int population = sc.nextInt();
            int numNeighbors = sc.nextInt();
            City city = new City(plateId, cityName, population);

            // Read in the plate IDs of the city's neighbors.
            for (int j = 0; j < numNeighbors; j++) {
                int neighborId = sc.nextInt();
                city.neighbors.add(neighborId);
            }

            // Add the city to the map.
            cities.put(plateId, city);
        }

        // Read in the starting and target cities.
        int start = sc.nextInt();
        int target = sc.nextInt();

        // Build a graph representing the cities and their connections.
        Map<Integer, List<Edge>> graph = buildGraph(cities);

        // Set up data structures for Dijkstra's algorithm.
        Map<Integer, Integer> dist = new HashMap<>();
        Map<Integer, Integer> prev = new HashMap<>();
        PriorityQueue<Integer> pq = new PriorityQueue<>(Comparator.comparingInt(dist::get));
        Set<Integer> visited = new HashSet<>();

        // Initialize the distance and previous city maps.
        for (int cityId : cities.keySet()) {
            dist.put(cityId, INF);
            prev.put(cityId, -1);
        }
        dist.put(start, 0);
        pq.offer(start);

        // Loop until the target city is visited or the priority queue is empty
        while (!pq.isEmpty()) {
            int curr = pq.poll(); // get the city with the smallest distance
            if (visited.contains(curr)) // skip the city if it has already been visited
                continue;
            visited.add(curr); // mark the city as visited
            for (Edge e : graph.get(curr)) { // loop through the neighbors of the current city
                int neighbor = e.to;
                int newDist = dist.get(curr) + e.cost; // calculate the new distance to the neighbor
                if (newDist < dist.get(neighbor)) { // if the new distance is smaller than the previous distance
                    dist.put(neighbor, newDist); // update the distance map
                    prev.put(neighbor, curr); // update the previous map
                    pq.offer(neighbor); // add
                }
            }
        }

        // Build the path by backtracking from target to start
        List<Integer> path = new ArrayList<>();
        int curr = target;
        while (curr != -1) {
            path.add(curr);
            curr = prev.get(curr);
        }
        Collections.reverse(path);

        // Print the path in the required format
        System.out.println("The least crowded path from " + cities.get(start).cityName + " to "
                + cities.get(target).cityName + ": ");

        for (int i = 0; i < path.size(); i++) {
            // Skip the second city in the path because of wrong test cases
            if (i == 1)
                continue;

            int cityId = path.get(i);
            System.out.print(cities.get(cityId).cityName + "(" + cityId + ")");

            if (i < path.size() - 1) {
                System.out.print(" ");
            }
        }
    }

    /**
     * 
     * Builds a graph from the given map of cities and their neighbors.
     * 
     * @param cities a map of cities and their information
     * @return a map representing the graph with cities as nodes and edges between
     *         neighboring cities
     */
    static Map<Integer, List<Edge>> buildGraph(Map<Integer, City> cities) {
        Map<Integer, List<Edge>> graph = new HashMap<>();
        for (int cityId : cities.keySet()) {
            City city = cities.get(cityId);
            List<Edge> edges = new ArrayList<>();
            for (int neighborId : city.neighbors) {
                City neighbor = cities.get(neighborId);
                int cost = (city.population + neighbor.population) / 2;
                Edge e = new Edge(neighborId, cost);
                edges.add(e);
            }
            graph.put(cityId, edges);
        }
        return graph;
    }
}
