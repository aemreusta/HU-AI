import java.io.Serializable;
import java.util.*; // For List, ArrayList, HashMap, PriorityQueue, Comparator, Set, HashSet, LinkedList

public class CampusNavigatorApp implements Serializable {
    static final long serialVersionUID = 99L;

    // These fields are declared but not used by the provided method signatures.
    // If needed for internal state/inspection, they can be populated.
    public HashMap<Station, Station> predecessors = new HashMap<>();
    public HashMap<Set<Station>, Double> times = new HashMap<>(); // This structure might be hard to use directly.

    public CampusNavigatorNetwork readCampusNavigatorNetwork(String filename) {
        CampusNavigatorNetwork network = new CampusNavigatorNetwork();
        // readInput can now throw RuntimeException on parsing or IO errors
        network.readInput(filename);
        return network;
    }

    private double calculateDistance(Point p1, Point p2) {
        if (p1 == null || p2 == null) {
             throw new IllegalArgumentException("Points cannot be null for distance calculation.");
        }
        return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
    }

    private double calculateWalkingTime(Station s1, Station s2, double walkingSpeedMpM) {
        if (s1 == null || s2 == null || s1.coordinates == null || s2.coordinates == null) {
            throw new IllegalArgumentException("Stations and their coordinates cannot be null for walking time calculation.");
        }
        if (walkingSpeedMpM <= 0) return Double.POSITIVE_INFINITY;
        double distance = calculateDistance(s1.coordinates, s2.coordinates);
        if (distance == 0) return 0.0; // Avoid division by zero if speed is huge and dist is tiny
        return distance / walkingSpeedMpM;
    }

    // OPTION 2: This method now expects cartSpeed in m/min directly
    private double calculateCartTime(Station s1, Station s2, double cartSpeedMpM_param) {
        if (s1 == null || s2 == null || s1.coordinates == null || s2.coordinates == null) {
            throw new IllegalArgumentException("Stations and their coordinates cannot be null for cart time calculation.");
        }
        if (cartSpeedMpM_param <= 0) return Double.POSITIVE_INFINITY;
        // No conversion needed here, cartSpeedMpM_param is already m/min
        double distance = calculateDistance(s1.coordinates, s2.coordinates);
        if (distance == 0) return 0.0;
        return distance / cartSpeedMpM_param;
    }

    public List<RouteDirection> getFastestRouteDirections(CampusNavigatorNetwork network) {
        // Basic validation of the network object
        if (network == null || network.startPoint == null || network.destinationPoint == null || network.lines == null) {
            return new ArrayList<>(); // Cannot proceed with incomplete network
        }

        Set<Station> uniqueNodes = new HashSet<>();
        uniqueNodes.add(network.startPoint);
        uniqueNodes.add(network.destinationPoint);
        for (CartLine line : network.lines) {
            if (line.cartLineStations != null) {
                uniqueNodes.addAll(line.cartLineStations);
            }
        }
        // If after parsing, startPoint or destinationPoint became null due to an error
        // caught and handled internally (not recommended, better to throw), uniqueNodes might miss them.
        // However, with readInput throwing exceptions on critical failures, this is less likely.
        
        List<Station> allNodes = new ArrayList<>(uniqueNodes);
        if (allNodes.isEmpty()) return new ArrayList<>(); // No nodes to process

        Map<Station, Double> dist = new HashMap<>();
        Map<Station, Station> prev = new HashMap<>();
        Map<Station, Boolean> prevEdgeWasCart = new HashMap<>(); // true if cart, false if walk

        for (Station station : allNodes) {
            dist.put(station, Double.POSITIVE_INFINITY);
        }
        
        // Ensure startPoint is in the dist map before trying to set its distance
        if (!dist.containsKey(network.startPoint)) {
            // This would happen if startPoint was null or not added to allNodes,
            // but the initial uniqueNodes.add(network.startPoint) should handle non-null startPoint.
            // If startPoint is null, the check at the beginning of the method should catch it.
            return new ArrayList<>(); // Cannot start if startPoint isn't a processable node
        }
        dist.put(network.startPoint, 0.0);

        PriorityQueue<Station> pq = new PriorityQueue<>(Comparator.comparingDouble(dist::get));
        pq.add(network.startPoint); // Add start node to begin Dijkstra's
        
        Set<Station> visited = new HashSet<>();

        double walkingSpeedMpM = network.averageWalkingSpeed; // This is m/min from network constant
        double cartSpeedMpM_from_network = network.averageCartSpeed; // This is m/min (as per Option 2)

        while (!pq.isEmpty()) {
            Station u = pq.poll();

            if (visited.contains(u)) {
                continue;
            }
            visited.add(u);

            if (u.equals(network.destinationPoint)) {
                break; // Found shortest path to destination
            }

            // Try walking from u to all other nodes v
            for (Station v : allNodes) {
                if (u.equals(v)) continue; // No self-loops for walking
                
                double walkTime = calculateWalkingTime(u, v, walkingSpeedMpM);
                // Check if u's distance is known (not infinity)
                if (dist.get(u) != null && dist.get(u) != Double.POSITIVE_INFINITY && dist.get(u) + walkTime < dist.get(v)) {
                    dist.put(v, dist.get(u) + walkTime);
                    prev.put(v, u);
                    prevEdgeWasCart.put(v, false); // Walk
                    pq.remove(v); // Update priority by re-adding
                    pq.add(v);
                }
            }

            // Try cart rides if u is a cart station
            for (CartLine line : network.lines) {
                List<Station> stationsOnLine = line.cartLineStations;
                if (stationsOnLine == null) continue;

                for (int i = 0; i < stationsOnLine.size(); i++) {
                    // Using .equals() for Station comparison.
                    // Assumes Station.equals() is either default (object identity, which is fine
                    // if each station object created during parsing is unique) or properly overridden.
                    // Given the context, object identity (==) is likely what's implicitly happening and intended.
                    if (stationsOnLine.get(i).equals(u)) { // u is station i on this line
                        // Check neighbor i-1 (previous station on line)
                        if (i > 0) {
                            Station vAdjacent = stationsOnLine.get(i - 1);
                            // Pass cartSpeedMpM_from_network (which is already m/min)
                            double cartTime = calculateCartTime(u, vAdjacent, cartSpeedMpM_from_network);
                            if (dist.get(u) != null && dist.get(u) != Double.POSITIVE_INFINITY && dist.get(u) + cartTime < dist.get(vAdjacent)) {
                                dist.put(vAdjacent, dist.get(u) + cartTime);
                                prev.put(vAdjacent, u);
                                prevEdgeWasCart.put(vAdjacent, true); // Cart
                                pq.remove(vAdjacent);
                                pq.add(vAdjacent);
                            }
                        }
                        // Check neighbor i+1 (next station on line)
                        if (i < stationsOnLine.size() - 1) {
                            Station vAdjacent = stationsOnLine.get(i + 1);
                            // Pass cartSpeedMpM_from_network (which is already m/min)
                            double cartTime = calculateCartTime(u, vAdjacent, cartSpeedMpM_from_network);
                             if (dist.get(u) != null && dist.get(u) != Double.POSITIVE_INFINITY && dist.get(u) + cartTime < dist.get(vAdjacent)) {
                                dist.put(vAdjacent, dist.get(u) + cartTime);
                                prev.put(vAdjacent, u);
                                prevEdgeWasCart.put(vAdjacent, true); // Cart
                                pq.remove(vAdjacent);
                                pq.add(vAdjacent);
                            }
                        }
                    }
                }
            }
        }
        
        // Reconstruct path
        LinkedList<RouteDirection> routeDirections = new LinkedList<>();
        Station current = network.destinationPoint;

        // Check if destination is reachable
        if (current == null || !dist.containsKey(current) || dist.get(current) == Double.POSITIVE_INFINITY) {
            return new ArrayList<>(); // Destination unreachable
        }
        
        // Handle case where start is destination
        boolean startIsDest = false;
        if (network.startPoint.equals(network.destinationPoint)) {
            // More robustly, if dist to destination is 0 AND there's no predecessor path.
             if (dist.get(network.destinationPoint) != null && dist.get(network.destinationPoint) == 0.0 && !prev.containsKey(network.destinationPoint)) {
                 startIsDest = true;
             }
        }

        if (!startIsDest) {
            while (prev.containsKey(current)) {
                Station predecessor = prev.get(current);
                if (predecessor == null) break; // Safety break, should not happen
                
                // Ensure current node is in prevEdgeWasCart map before accessing
                if (!prevEdgeWasCart.containsKey(current)) {
                    // This indicates an issue in path reconstruction or map population
                    break; // Or throw an error
                }
                boolean isCartRide = prevEdgeWasCart.get(current);
                double duration;
                if (isCartRide) {
                    duration = calculateCartTime(predecessor, current, cartSpeedMpM_from_network);
                } else {
                    duration = calculateWalkingTime(predecessor, current, walkingSpeedMpM);
                }
                routeDirections.addFirst(new RouteDirection(predecessor.description, current.description, duration, isCartRide));
                current = predecessor;
                if (current.equals(network.startPoint)) {
                    break; // Path reconstruction complete
                }
            }
        }
        
        return new ArrayList<>(routeDirections);
    }

    /**
     * Function to print the route directions to STDOUT
     */
    public void printRouteDirections(List<RouteDirection> directions) {
        double totalDuration = 0;
        if (directions == null) { // Should be an empty list if no path, not null
            directions = new ArrayList<>();
        }

        for (RouteDirection dir : directions) {
            totalDuration += dir.duration;
        }

        long roundedTotalTime = Math.round(totalDuration);
        System.out.println("The fastest route takes " + roundedTotalTime + " minute(s).");

        if (!directions.isEmpty()) {
            System.out.println("Directions");
            System.out.println("----------");
            int stepNumber = 1;
            for (RouteDirection dir : directions) {
                String mode = dir.cartRide ? "Ride the cart" : "Walk";
                System.out.printf(Locale.US, "%d. %s from \"%s\" to \"%s\" for %.2f minutes.%n",
                        stepNumber++,
                        mode,
                        dir.startStationName,
                        dir.endStationName,
                        dir.duration);
            }
        }
    }
}