import java.io.Serializable;
import java.util.*;

public class CampusNavigatorApp implements Serializable {
    static final long serialVersionUID = 99L;

    // These might be better placed inside getFastestRouteDirections or as part of a graph class
    // Making them transient as they likely represent state for a single calculation
    private transient Map<Station, Station> predecessors;
    private transient Map<Station, Double> distances; // Renamed from 'times' for clarity in Dijkstra

    public CampusNavigatorNetwork readCampusNavigatorNetwork(String filename) {
        CampusNavigatorNetwork network = new CampusNavigatorNetwork();
        network.readInput(filename);
        return network;
    }

    // Helper method to calculate Euclidean distance
    private double calculateDistance(Point p1, Point p2) {
        return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
    }

    // Helper class for Priority Queue in Dijkstra
    private static class StationDistance implements Comparable<StationDistance> {
        Station station;
        double distance;

        StationDistance(Station station, double distance) {
            this.station = station;
            this.distance = distance;
        }

        @Override
        public int compareTo(StationDistance other) {
            return Double.compare(this.distance, other.distance);
        }
         // Need equals/hashCode if used in Sets etc. Based on Station.
         @Override
         public boolean equals(Object o) {
              if (this == o) return true;
              if (o == null || getClass() != o.getClass()) return false;
              StationDistance that = (StationDistance) o;
              return Objects.equals(station, that.station); // Equality based on station only for PQ updates
         }
         @Override
         public int hashCode() {
             return Objects.hash(station);
         }
    }

    // Helper to store edge type (walk/cart) during Dijkstra
    private transient Map<Station, Boolean> edgeTypeToReachStation; // Key: destination station, Value: true if reached by cart

    /**
     * Calculates the fastest route from the user's selected starting point to
     * the desired destination, using the campus golf cart network and walking paths.
     * Uses Dijkstra's algorithm.
     * @return List of RouteDirection instances, or empty list if no path found.
     */
    public List<RouteDirection> getFastestRouteDirections(CampusNavigatorNetwork network) {
        predecessors = new HashMap<>();
        distances = new HashMap<>();
        edgeTypeToReachStation = new HashMap<>(); // Track how we reached each node

        // Check for null network or points
        if (network == null || network.startPoint == null || network.destinationPoint == null) {
             System.err.println("Error: Network or start/destination points are null.");
             return new ArrayList<>();
        }

        // 1. Collect all unique stations (nodes)
        Set<Station> allStations = new HashSet<>();
        allStations.add(network.startPoint);
        allStations.add(network.destinationPoint);
        if (network.lines != null) {
            for (CartLine line : network.lines) {
                if (line.cartLineStations != null) {
                    allStations.addAll(line.cartLineStations);
                }
            }
        }

        // 2. Initialize Dijkstra
        for (Station s : allStations) {
            distances.put(s, Double.POSITIVE_INFINITY);
            predecessors.put(s, null);
        }
        distances.put(network.startPoint, 0.0);

        PriorityQueue<StationDistance> pq = new PriorityQueue<>();
        pq.add(new StationDistance(network.startPoint, 0.0));

        Set<Station> visited = new HashSet<>(); // Keep track of visited nodes

        // Cart speed in meters per minute
        double cartSpeed_mpm = network.getCartSpeedMetersPerMinute();
        // Walking speed in meters per minute
        double walkSpeed_mpm = network.averageWalkingSpeed_mpm; // Defined in network

        // 3. Run Dijkstra
        while (!pq.isEmpty()) {
            StationDistance current = pq.poll();
            Station u = current.station;

            // Avoid reprocessing if a shorter path was already found
             if (visited.contains(u) || current.distance > distances.get(u)) {
                 continue;
            }
             visited.add(u);

            // If we reached the destination, we can potentially stop (if graph has non-negative weights)
            // However, processing all reachable nodes ensures correctness even with complex paths.
             // if (u.equals(network.destinationPoint)) {
             //     break; // Optimization possible here
             // }


            // Explore neighbors:

            // a) Walking neighbors (all other stations)
            for (Station v : allStations) {
                if (u.equals(v)) continue; // Skip self

                double dist_uv = calculateDistance(u.coordinates, v.coordinates);
                double time_walk = (walkSpeed_mpm > 0) ? (dist_uv / walkSpeed_mpm) : Double.POSITIVE_INFINITY;
                double newDist = distances.get(u) + time_walk;

                if (newDist < distances.get(v)) {
                    distances.put(v, newDist);
                    predecessors.put(v, u);
                    edgeTypeToReachStation.put(v, false); // Reached by walking
                    pq.add(new StationDistance(v, newDist));
                }
            }

            // b) Cart neighbors (only if u is part of a cart line)
             if (network.lines != null && cartSpeed_mpm > 0) {
                 for (CartLine line : network.lines) {
                     List<Station> stations = line.cartLineStations;
                     if (stations == null) continue;
                     for (int i = 0; i < stations.size(); i++) {
                         if (stations.get(i).equals(u)) { // Found station u on this line
                             // Check connection to previous station (if exists)
                             if (i > 0) {
                                 Station v_prev = stations.get(i - 1);
                                 double dist_cart = calculateDistance(u.coordinates, v_prev.coordinates);
                                 double time_cart = dist_cart / cartSpeed_mpm;
                                 double newDistCart = distances.get(u) + time_cart;

                                 if (newDistCart < distances.get(v_prev)) {
                                     distances.put(v_prev, newDistCart);
                                     predecessors.put(v_prev, u);
                                     edgeTypeToReachStation.put(v_prev, true); // Reached by cart
                                     pq.add(new StationDistance(v_prev, newDistCart));
                                 }
                             }
                             // Check connection to next station (if exists)
                             if (i < stations.size() - 1) {
                                 Station v_next = stations.get(i + 1);
                                 double dist_cart = calculateDistance(u.coordinates, v_next.coordinates);
                                 double time_cart = dist_cart / cartSpeed_mpm;
                                 double newDistCart = distances.get(u) + time_cart;

                                 if (newDistCart < distances.get(v_next)) {
                                     distances.put(v_next, newDistCart);
                                     predecessors.put(v_next, u);
                                     edgeTypeToReachStation.put(v_next, true); // Reached by cart
                                     pq.add(new StationDistance(v_next, newDistCart));
                                 }
                             }
                         }
                     }
                 }
             }
        }

        // 4. Reconstruct path
        List<RouteDirection> routeDirections = new LinkedList<>(); // Use LinkedList for efficient adding at start
        Station step = network.destinationPoint;

        // Check if destination is reachable
        if (predecessors.get(step) == null && !step.equals(network.startPoint)) {
            System.out.println("Destination is not reachable.");
            return new ArrayList<>(); // Return empty list
        }

        while (predecessors.get(step) != null) {
            Station prevStep = predecessors.get(step);
            boolean isCartRide = edgeTypeToReachStation.getOrDefault(step, false); // Check how 'step' was reached

            // Recalculate the time for this specific step to avoid potential floating point accumulations
            double stepDistance = calculateDistance(prevStep.coordinates, step.coordinates);
            double stepDuration;
            if (isCartRide) {
                 // Verify this was indeed a cart segment
                 boolean actuallyCart = false;
                 if(network.lines != null && cartSpeed_mpm > 0) {
                      for(CartLine line : network.lines) {
                           List<Station> lineStations = line.cartLineStations;
                           if(lineStations == null) continue;
                           for(int i = 0; i < lineStations.size() - 1; i++) {
                                if((lineStations.get(i).equals(prevStep) && lineStations.get(i+1).equals(step)) ||
                                   (lineStations.get(i).equals(step) && lineStations.get(i+1).equals(prevStep))) {
                                     actuallyCart = true;
                                     break;
                                }
                           }
                           if(actuallyCart) break;
                      }
                 }
                 // If the predecessor map indicated cart, but we can't find it, assume walk? Or recalculate based on flags.
                 // Sticking to the flag for now.
                 stepDuration = (cartSpeed_mpm > 0) ? (stepDistance / cartSpeed_mpm) : Double.POSITIVE_INFINITY;

            } else {
                stepDuration = (walkSpeed_mpm > 0) ? (stepDistance / walkSpeed_mpm) : Double.POSITIVE_INFINITY;
            }


            // Use the duration calculated during Dijkstra for this segment: distances.get(step) - distances.get(prevStep)
            // This might be more robust against recalculation issues if edge weights were complex. Let's use this.
            double durationForStep = distances.get(step) - distances.get(prevStep);

            routeDirections.add(0, new RouteDirection(prevStep.description, step.description, durationForStep, isCartRide));
            step = prevStep;
        }

        //Collections.reverse(routeDirections); // Already added at index 0, so it's in correct order

        return routeDirections;
    }


    /**
     * Function to print the route directions to STDOUT in the specified format.
     */
    public void printRouteDirections(List<RouteDirection> directions) {
        if (directions == null || directions.isEmpty()) {
            // Handle case where no route was found or directions list is null
            System.out.println("No route found or directions are unavailable.");
             // Print the required headers even if no route exists? The example implies a route exists.
             // Let's assume directions will not be null/empty if a route exists as per example.
             // If it could be empty, print a specific message.
            return;
        }

        double totalDuration = 0;
        for (RouteDirection dir : directions) {
            totalDuration += dir.duration;
        }

        // Round total duration to the nearest minute
        long roundedTotalMinutes = Math.round(totalDuration);

        System.out.printf(Locale.US,"The fastest route takes %d minute(s).%n", roundedTotalMinutes);
        System.out.println("Directions");
        System.out.println("----------");

        int stepNumber = 1;
        for (RouteDirection dir : directions) {
            String travelType = dir.cartRide ? "Ride the cart" : "Walk";
            // Print step duration with exactly two decimal places
            System.out.printf(Locale.US, "%d. %s from \"%s\" to \"%s\" for %.2f minutes.%n",
                    stepNumber++,
                    travelType,
                    dir.startStationName,
                    dir.endStationName,
                    dir.duration);
        }
    }
}