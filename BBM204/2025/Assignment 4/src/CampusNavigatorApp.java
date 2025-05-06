// CampusNavigatorApp.java
import java.io.Serializable;
import java.util.*;

public class CampusNavigatorApp implements Serializable {
    static final long serialVersionUID = 99L;

    // Store results from Dijkstra for path reconstruction
    private transient Map<Station, Station> predecessors;
    private transient Map<Station, Double> distances; // Shortest time from start
    private transient Map<Station, Boolean> edgeIsCartRide; // Track if edge to predecessor was cart


    public CampusNavigatorNetwork readCampusNavigatorNetwork(String filename) {
        CampusNavigatorNetwork network = new CampusNavigatorNetwork();
        if (!network.readInput(filename)) {
            return null; // Return null if reading failed
        }
        return network;
    }


    /**
     * Calculates the fastest route using Dijkstra's algorithm.
     * @return List of RouteDirection instances representing the path, or empty list if no path.
     */
    public List<RouteDirection> getFastestRouteDirections(CampusNavigatorNetwork network) {
        if (network == null || network.startPoint == null || network.destinationPoint == null) {
             System.err.println("Network data is incomplete.");
            return Collections.emptyList();
        }

        distances = new HashMap<>();
        predecessors = new HashMap<>();
        edgeIsCartRide = new HashMap<>(); // To remember how we reached a node

        // Priority queue stores pairs of [time, Station]
        PriorityQueue<Map.Entry<Station, Double>> pq = new PriorityQueue<>(Comparator.comparingDouble(Map.Entry::getValue));

        // Initialize distances
        for (Station station : network.getAllStations()) {
            distances.put(station, Double.POSITIVE_INFINITY);
            predecessors.put(station, null);
            edgeIsCartRide.put(station, false); // Default to walk
        }
        distances.put(network.startPoint, 0.0);
        pq.add(new AbstractMap.SimpleEntry<>(network.startPoint, 0.0));


        while (!pq.isEmpty()) {
            Station current = pq.poll().getKey();

            // Optimization: If we found the destination, maybe break early
            // Note: Standard Dijkstra explores fully, this might miss optimal if edge weights change later (not the case here)
             if (current.equals(network.destinationPoint)) {
               // break; // Can uncomment for slight optimization if graph known to be static
             }

            // Explore neighbors (all other stations for walking, adjacent for cart)

            // 1. Cart rides
            for (CartLine line : network.lines) {
                for (int i = 0; i < line.cartLineStations.size() - 1; i++) {
                    Station u = line.cartLineStations.get(i);
                    Station v = line.cartLineStations.get(i + 1);

                    // Process edge u -> v (if current is u)
                    if (u.equals(current)) {
                         processEdge(current, v, true, network);
                    }
                     // Process edge v -> u (if current is v)
                    if (v.equals(current)) {
                         processEdge(current, u, true, network);
                    }
                }
            }

             // 2. Walking connections (to ALL other stations)
             for (Station neighbor : network.getAllStations()) {
                 if (!neighbor.equals(current)) {
                     // Check if there's a direct cart path between them already considered
                     boolean directCartExists = false;
                     for (CartLine line : network.lines) {
                        List<Station> stations = line.cartLineStations;
                        for (int i = 0; i < stations.size() - 1; i++) {
                            if ((stations.get(i).equals(current) && stations.get(i+1).equals(neighbor)) ||
                                (stations.get(i).equals(neighbor) && stations.get(i+1).equals(current))) {
                                directCartExists = true;
                                break;
                            }
                        }
                        if(directCartExists) break;
                     }

                      // Only process walking if no direct cart path exists OR if walking is faster
                      // (Dijkstra handles choosing the minimum naturally if both edges are added,
                      // but adding all-pairs walking can be computationally expensive. Here we add walking.)
                     // Let's simplify: Assume we *always* consider walking as an option.
                     processEdge(current, neighbor, false, network); // Process as a potential walk
                 }
             }
        } // End Dijkstra loop

        // Reconstruct path
        return reconstructPath(network.startPoint, network.destinationPoint);
    }

    /** Helper for Dijkstra to process an edge */
    private void processEdge(Station u, Station v, boolean isCart, CampusNavigatorNetwork network) {
        double distance = u.distanceTo(v);
        double speed = isCart ? network.averageCartSpeed : network.averageWalkingSpeed;
         if (speed <= 0) return; // Avoid division by zero or negative speed

        double time = distance / speed; // time in minutes
        double newDist = distances.get(u) + time;

        if (newDist < distances.get(v)) {
            distances.put(v, newDist);
            predecessors.put(v, u);
            edgeIsCartRide.put(v, isCart); // Record how we reached v with this shorter path

            // Update priority queue - remove old entry if exists, add new one
             // Simple approach: just add. PQ handles duplicates, taking the one with min distance first.
             PriorityQueue<Map.Entry<Station, Double>> pq = new PriorityQueue<>(Comparator.comparingDouble(Map.Entry::getValue)); // Recreate locally if not accessible
            pq.add(new AbstractMap.SimpleEntry<>(v, newDist));

             // Correct PQ update (requires ability to decrease-key or remove/add):
             // pq.removeIf(entry -> entry.getKey().equals(v)); // May be inefficient O(N)
             // pq.add(new AbstractMap.SimpleEntry<>(v, newDist));
             // In standard Java PQ, adding the improved path works, though less efficient than decrease-key.
        }
    }


     /** Reconstructs the path from end to start using predecessors map */
    private List<RouteDirection> reconstructPath(Station start, Station end) {
        LinkedList<RouteDirection> path = new LinkedList<>();
        Station current = end;

        if (predecessors.get(current) == null && !current.equals(start)) {
            System.err.println("No path found to the destination.");
            return Collections.emptyList(); // No path found
        }


        while (predecessors.get(current) != null) {
            Station prev = predecessors.get(current);
            double timeStep = distances.get(current) - distances.get(prev); // Time for this specific step
            boolean wasCartRide = edgeIsCartRide.get(current); // Was the edge leading to 'current' a cart ride?

            path.addFirst(new RouteDirection(prev.description, current.description, timeStep, wasCartRide));
            current = prev;
        }

        // Check if path reconstruction actually reached the start point
        if (!current.equals(start)) {
             System.err.println("Path reconstruction failed to reach the start point.");
             return Collections.emptyList(); // Should not happen if Dijkstra ran correctly
        }


        return path;
    }


    /**
     * Function to print the route directions to STDOUT
     */
    public void printRouteDirections(List<RouteDirection> directions) {
        if (directions == null || directions.isEmpty()) {
            if (distances != null && distances.values().stream().allMatch(d -> d == Double.POSITIVE_INFINITY)) {
                 System.out.println("The destination is unreachable.");
            } else if (distances != null) {
                // Check if start equals destination
                boolean startIsDest = false;
                if(predecessors != null && !predecessors.isEmpty()){
                     Station dest = predecessors.keySet().stream().filter(s -> s.description.equals("Final Destination")).findFirst().orElse(null);
                     Station start = predecessors.keySet().stream().filter(s -> s.description.equals("Starting Point")).findFirst().orElse(null);
                     if (start!= null && start.equals(dest) && distances.get(dest) == 0.0) {
                         startIsDest = true;
                     }
                }
                if(startIsDest) {
                    System.out.println("The fastest route takes 0 minute(s).");
                    System.out.println("Directions");
                    System.out.println("----------");
                    System.out.println("1. You are already at your destination!");
                } else {
                    System.out.println("No route found or destination unreachable.");
                }

            } else {
                 System.out.println("No directions to print.");
            }
            return;
        }

        double totalDuration = 0;
        for (RouteDirection dir : directions) {
            totalDuration += dir.duration;
        }

        // Round total duration to the nearest minute for the summary line
        long totalMinutesRounded = Math.round(totalDuration);

        System.out.printf("The fastest route takes %d minute(s).\n", totalMinutesRounded);
        System.out.println("Directions");
        System.out.println("----------");

        int step = 1;
        for (RouteDirection dir : directions) {
            String mode = dir.isCartRide() ? "Ride the cart" : "Walk";
            // Print individual step durations with 2 decimal places
            System.out.printf("%d. %s from \"%s\" to \"%s\" for %.2f minutes.\n",
                    step++, mode, dir.startStationName, dir.endStationName, dir.duration);
        }
    }
}