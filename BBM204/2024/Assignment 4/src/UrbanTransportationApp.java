import java.io.Serializable;
import java.util.*;

class UrbanTransportationApp implements Serializable {
    static final long serialVersionUID = 99L;

    public HyperloopTrainNetwork readHyperloopTrainNetwork(String filename) {
        HyperloopTrainNetwork hyperloopTrainNetwork = new HyperloopTrainNetwork();
        hyperloopTrainNetwork.readInput(filename);
        return hyperloopTrainNetwork;
    }

    public List<RouteDirection> getFastestRouteDirections(HyperloopTrainNetwork network) {
        List<RouteDirection> routeDirections = new ArrayList<>();
        Map<Station, Double> times = new HashMap<>();
        Map<Station, Station> previous = new HashMap<>();
        Map<Station, RouteDirection> directionMap = new HashMap<>();
        PriorityQueue<Station> queue = new PriorityQueue<>(Comparator.comparingDouble(times::get));

        // Print train speed and walking speed
        // System.out.println("Train speed: " + network.averageTrainSpeed + " m/min");
        // System.out.println("Walking speed: " + network.averageWalkingSpeed + " m/min");

        // Initialize start point
        times.put(network.startPoint, 0.0);
        queue.add(network.startPoint);

        while (!queue.isEmpty()) {
            Station current = queue.poll();
            double currentTime = times.get(current);

            // System.out.println("Visiting station: " + current.description + " with current time: " + currentTime);

            // If we reached the destination, break the loop
            if (current.equals(network.destinationPoint)) {
                // System.out.println("Reached destination: " + current.description);
                break;
            }

            // Check train rides from current station
            for (TrainLine line : network.lines) {
                if (line.trainLineStations.contains(current)) {
                    int currentIndex = line.trainLineStations.indexOf(current);
                    for (int i = 0; i < line.trainLineStations.size(); i++) {
                        if (i != currentIndex) {
                            Station neighbor = line.trainLineStations.get(i);
                            double trainTime = calculateTrainTime(current.coordinates, neighbor.coordinates, network.averageTrainSpeed);
                            double newTime = currentTime + trainTime;
                            if (newTime < times.getOrDefault(neighbor, Double.MAX_VALUE)) {
                                times.put(neighbor, newTime);
                                previous.put(neighbor, current);
                                directionMap.put(neighbor, new RouteDirection(current.description, neighbor.description, trainTime, true));
                                queue.add(neighbor);
                                // System.out.println("Adding neighbor via train: " + neighbor.description + " with time: " + newTime);
                            }
                        }
                    }
                }
            }

            // Check walking to other stations
            for (TrainLine line : network.lines) {
                for (Station neighbor : line.trainLineStations) {
                    if (!neighbor.equals(current)) {
                        double walkTime = calculateWalkTime(current.coordinates, neighbor.coordinates, network.averageWalkingSpeed);
                        double newTime = currentTime + walkTime;
                        if (newTime < times.getOrDefault(neighbor, Double.MAX_VALUE)) {
                            times.put(neighbor, newTime);
                            previous.put(neighbor, current);
                            directionMap.put(neighbor, new RouteDirection(current.description, neighbor.description, walkTime, false));
                            queue.add(neighbor);
                            // System.out.println("Adding neighbor via walking: " + neighbor.description + " with time: " + newTime);
                        }
                    }
                }
            }
        }

        // Add the walk from the last station to the final destination if needed
        if (!times.containsKey(network.destinationPoint)) {
            Station nearestStation = null;
            double minTime = Double.MAX_VALUE;
            for (Station station : times.keySet()) {
                double walkTime = calculateWalkTime(station.coordinates, network.destinationPoint.coordinates, network.averageWalkingSpeed);
                double totalTime = times.get(station) + walkTime;
                if (totalTime < minTime) {
                    nearestStation = station;
                    minTime = totalTime;
                }
            }
            if (nearestStation != null) {
                times.put(network.destinationPoint, minTime);
                previous.put(network.destinationPoint, nearestStation);
                directionMap.put(network.destinationPoint, new RouteDirection(nearestStation.description, "Final Destination", minTime - times.get(nearestStation), false));
                // System.out.println("Adding walk to final destination from: " + nearestStation.description + " with time: " + minTime);
            }
        }

        // Reconstruct path
        List<Station> path = new ArrayList<>();
        for (Station at = network.destinationPoint; at != null; at = previous.get(at)) {
            path.add(at);
        }
        Collections.reverse(path);

        // Convert path to route directions
        for (int i = 0; i < path.size() - 1; i++) {
            Station start = path.get(i);
            Station end = path.get(i + 1);
            routeDirections.add(directionMap.get(end));
        }

        // Print the final path for debugging
        // System.out.println("Final path:");
        // for (Station station : path) {
        //     System.out.println(station.description + " at (" + station.coordinates.x + ", " + station.coordinates.y + ")");
        // }

        return routeDirections;
    }

    private double calculateTrainTime(Point start, Point end, double trainSpeed) {
        double distance = Math.sqrt(Math.pow(end.x - start.x, 2) + Math.pow(end.y - start.y, 2));
        return distance / trainSpeed;
    }

    private double calculateWalkTime(Point start, Point end, double walkingSpeed) {
        double distance = Math.sqrt(Math.pow(end.x - start.x, 2) + Math.pow(end.y - start.y, 2));
        return distance / walkingSpeed;
    }

    public void printRouteDirections(List<RouteDirection> directions) {
        double totalDuration = directions.stream().mapToDouble(d -> d.duration).sum();
        System.out.println("The fastest route takes " + Math.round(totalDuration) + " minute(s).");
        System.out.println("Directions");
        System.out.println("----------");
        for (int i = 0; i < directions.size(); i++) {
            RouteDirection dir = directions.get(i);
            String mode = dir.trainRide ? "Get on the train" : "Walk";
            System.out.printf("%d. %s from \"%s\" to \"%s\" for %.2f minutes.%n",
                    i + 1, mode, dir.startStationName, dir.endStationName, dir.duration);
        }
    }
}
