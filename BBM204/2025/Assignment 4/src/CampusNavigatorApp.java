import java.io.Serializable;
import java.util.*;
import java.text.DecimalFormat; // For formatting output

public class CampusNavigatorApp implements Serializable {
    static final long serialVersionUID = 99L;

    // Removed these as they are better scoped within getFastestRouteDirections
    // public HashMap<Station, Station> predecessors = new HashMap<>();
    // public HashMap<Set<Station>, Double> times = new HashMap<>();

    public CampusNavigatorNetwork readCampusNavigatorNetwork(String filename) {
        CampusNavigatorNetwork network = new CampusNavigatorNetwork();
        network.readInput(filename); // This might lead to an incompletely initialized network if file ops fail
        return network;
    }

    private double calculateDistance(Point p1, Point p2) {
        if (p1 == null || p2 == null) return Double.POSITIVE_INFINITY; // Safety
        return Math.hypot(p1.x - p2.x, p1.y - p2.y);
    }
    
    // Helper to find a station in a list by its description (for robust predecessor tracking)
    private Station findStationByDescription(String description, List<Station> nodes) {
        for (Station s : nodes) {
            if (s.description.equals(description)) {
                return s;
            }
        }
        return null; 
    }


    public List<RouteDirection> getFastestRouteDirections(CampusNavigatorNetwork network) {
        List<RouteDirection> routeDirections = new ArrayList<>();

        // Graceful handling if network is not properly initialized
        if (network == null || network.startPoint == null || network.destinationPoint == null || network.lines == null) {
            // System.err.println("Network not properly initialized for pathfinding.");
            return routeDirections; // Return empty list
        }
        
        // Convert speeds to meters per minute
        double walkingSpeedMpM = network.averageWalkingSpeed; 
        double cartSpeedMpM = network.averageCartSpeed * 1000.0 / 60.0;

        List<Station> allNodes = new ArrayList<>();
        // Add start and end points first
        allNodes.add(network.startPoint);
        // Ensure destination is distinct from start before adding, if they could be same object by chance
        if (!network.startPoint.description.equals(network.destinationPoint.description) ||
            network.startPoint.coordinates.x != network.destinationPoint.coordinates.x ||
            network.startPoint.coordinates.y != network.destinationPoint.coordinates.y) {
            allNodes.add(network.destinationPoint);
        }


        for (CartLine line : network.lines) {
            if (line.cartLineStations == null) continue;
            for (Station stationFromLine : line.cartLineStations) {
                boolean exists = false;
                for(Station existingNode : allNodes) {
                    if(existingNode.description.equals(stationFromLine.description) && 
                       existingNode.coordinates.x == stationFromLine.coordinates.x &&
                       existingNode.coordinates.y == stationFromLine.coordinates.y) {
                        exists = true;
                        break;
                    }
                }
                if(!exists) {
                    allNodes.add(stationFromLine);
                }
            }
        }
        
        Map<Station, Double> minDist = new HashMap<>();
        Map<Station, RouteDirection> predecessorEdge = new HashMap<>();
        // Use PriorityQueue with a comparator for Station entries
        PriorityQueue<Map.Entry<Station, Double>> pq = new PriorityQueue<>(Map.Entry.comparingByValue());

        for (Station node : allNodes) {
            minDist.put(node, Double.POSITIVE_INFINITY);
        }

        // Check if startPoint is actually in allNodes (it should be)
        Station dijkstraStartNode = findStationByDescription(network.startPoint.description, allNodes);
        if (dijkstraStartNode == null) { // Should not happen if allNodes is populated correctly
            // System.err.println("Dijkstra start node not found in allNodes list.");
            return routeDirections;
        }

        minDist.put(dijkstraStartNode, 0.0);
        pq.add(new AbstractMap.SimpleEntry<>(dijkstraStartNode, 0.0));

        Set<Station> settledNodes = new HashSet<>();

        while (!pq.isEmpty()) {
            Map.Entry<Station, Double> currentEntry = pq.poll();
            Station u = currentEntry.getKey();
            double u_dist = currentEntry.getValue();

            // If already found a shorter path to u or u is settled with shorter path
            if (u_dist > minDist.getOrDefault(u, Double.POSITIVE_INFINITY) || settledNodes.contains(u)) {
                continue;
            }
            settledNodes.add(u);

            // Optimization: if u is the destination, we can stop if graph has non-negative edges (which it does)
            if (u.description.equals(network.destinationPoint.description)) {
                 // For some Dijkstra variants, one might break here.
                 // Continue to explore all paths to ensure optimality if there are tricky parts, but typically safe.
            }


            // Consider walking to all other nodes (v)
            for (Station v : allNodes) {
                if (u.equals(v)) continue;

                double distUV_walk = calculateDistance(u.coordinates, v.coordinates);
                double timeUV_walk = distUV_walk / walkingSpeedMpM;
                
                if (minDist.get(u) + timeUV_walk < minDist.getOrDefault(v, Double.POSITIVE_INFINITY)) {
                    minDist.put(v, minDist.get(u) + timeUV_walk);
                    predecessorEdge.put(v, new RouteDirection(u.description, v.description, timeUV_walk, false));
                    pq.removeIf(entry -> entry.getKey().description.equals(v.description)); // Remove old entry if exists
                    pq.add(new AbstractMap.SimpleEntry<>(v, minDist.get(v)));
                }
            }

            // Consider cart rides from station u (if u is a cart station)
            for (CartLine line : network.lines) {
                if (line.cartLineStations == null) continue;
                List<Station> stationsOnLine = line.cartLineStations;
                int u_line_idx = -1;
                for(int i=0; i<stationsOnLine.size(); i++){
                    Station s = stationsOnLine.get(i);
                    if(s.description.equals(u.description) && s.coordinates.x == u.coordinates.x && s.coordinates.y == u.coordinates.y){
                        u_line_idx = i;
                        break;
                    }
                }

                if (u_line_idx != -1) { // u is a station on this cart line
                    Station u_on_line = stationsOnLine.get(u_line_idx); // The actual object from cartLineStations
                    // Ride to previous station
                    if (u_line_idx > 0) {
                        Station prevStationOnLine = stationsOnLine.get(u_line_idx - 1);
                        // Find corresponding Station object in allNodes
                        Station v_target = findStationByDescription(prevStationOnLine.description, allNodes);
                        if (v_target != null) {
                            double distCart = calculateDistance(u_on_line.coordinates, v_target.coordinates);
                            double timeCart = distCart / cartSpeedMpM;
                            if (minDist.get(u) + timeCart < minDist.getOrDefault(v_target, Double.POSITIVE_INFINITY)) {
                                minDist.put(v_target, minDist.get(u) + timeCart);
                                predecessorEdge.put(v_target, new RouteDirection(u.description, v_target.description, timeCart, true));
                                pq.removeIf(entry -> entry.getKey().description.equals(v_target.description));
                                pq.add(new AbstractMap.SimpleEntry<>(v_target, minDist.get(v_target)));
                            }
                        }
                    }
                    // Ride to next station
                    if (u_line_idx < stationsOnLine.size() - 1) {
                        Station nextStationOnLine = stationsOnLine.get(u_line_idx + 1);
                        Station v_target = findStationByDescription(nextStationOnLine.description, allNodes);
                        if (v_target != null) {
                            double distCart = calculateDistance(u_on_line.coordinates, v_target.coordinates);
                            double timeCart = distCart / cartSpeedMpM;
                            if (minDist.get(u) + timeCart < minDist.getOrDefault(v_target, Double.POSITIVE_INFINITY)) {
                                minDist.put(v_target, minDist.get(u) + timeCart);
                                predecessorEdge.put(v_target, new RouteDirection(u.description, v_target.description, timeCart, true));
                                pq.removeIf(entry -> entry.getKey().description.equals(v_target.description));
                                pq.add(new AbstractMap.SimpleEntry<>(v_target, minDist.get(v_target)));
                            }
                        }
                    }
                }
            }
        }

        // Reconstruct path
        Station destinationNode = findStationByDescription(network.destinationPoint.description, allNodes);
        if (destinationNode == null || !predecessorEdge.containsKey(destinationNode)) {
             // No path found to destination or destination is start.
             // System.err.println("No path found to destination or destination is start point and not processed.");
        } else {
            Station current = destinationNode;
            while (predecessorEdge.containsKey(current)) {
                RouteDirection edge = predecessorEdge.get(current);
                routeDirections.add(edge);
                Station predecessor = findStationByDescription(edge.startStationName, allNodes);
                if (predecessor == null || predecessor.description.equals(current.description)) { // Break if loop or no pred
                    // System.err.println("Error in path reconstruction or start reached.");
                    break; 
                }
                current = predecessor;
                if (current.description.equals(network.startPoint.description)) break;
            }
        }
        Collections.reverse(routeDirections);
        return routeDirections;
    }


    public void printRouteDirections(List<RouteDirection> directions) {
        double totalDuration = 0;
        if (directions == null ) { // Should not be null, but defensive
            directions = new ArrayList<>();
        }

        for (RouteDirection dir : directions) {
            totalDuration += dir.duration;
        }

        System.out.println(String.format("The fastest route takes %d minute(s).", Math.round(totalDuration)));
        System.out.println("Directions");
        System.out.println("----------");

        if (directions.isEmpty() && totalDuration == 0) {
             // No explicit instruction for "no route" or "start is destination" if total time is 0.
             // The sample output always shows a path, so if directions list is empty,
             // it might imply start=destination or an error.
             // If totalDuration > 0 but directions is empty, that's an issue.
        } else {
            DecimalFormat df = new DecimalFormat("0.00"); // Ensure English locale for dot.
            df.setRoundingMode(java.math.RoundingMode.HALF_UP); // Standard rounding
            int step = 1;
            for (RouteDirection dir : directions) {
                String travelMode = dir.cartRide ? "Ride the cart" : "Walk";
                System.out.println(String.format("%d. %s from \"%s\" to \"%s\" for %s minutes.",
                        step++,
                        travelMode,
                        dir.startStationName,
                        dir.endStationName,
                        df.format(dir.duration) // Use df.format for consistent two decimal places
                ));
            }
        }
    }
}