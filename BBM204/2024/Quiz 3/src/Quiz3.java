import java.util.*;
import java.io.*;

public class Quiz3 {

  public static void main(String[] args) throws IOException {
    Scanner sc = new Scanner(new File("/Users/emre/GitHub/HU-AI/BBM204/2024/Quiz 3/src/sample_io/sample_input_0.txt"));

    int T = sc.nextInt(); // Number of test cases

    for (int t = 0; t < T; t++) {
        int S = sc.nextInt(); // Stations with drones
        int P = sc.nextInt(); // Total stations

        // Create graph representation
        Map<Integer, List<Edge>> graph = new HashMap<>();
        for (int i = 1; i <= P; i++) {
            int x = sc.nextInt();
            int y = sc.nextInt();
            graph.put(i, new ArrayList<>());
            graph.get(i).add(new Edge(i, 0.0, x, y));  // Assuming weight is 0.0 for initial station
        }


        // Add edges based on drone availability and distance
        for (int i = 1; i <= P; i++) {
            for (int j = i + 1; j <= P; j++) {
                double distance = Math.sqrt(Math.pow(graph.get(i).get(0).x - graph.get(j).get(0).x, 2) +
                                            Math.pow(graph.get(i).get(0).y - graph.get(j).get(0).y, 2));
                if (hasDrone(i) && hasDrone(j) || hasDrone(i) || hasDrone(j)) {
                    // Both have drones or one has a drone (unlimited range)
                    graph.get(i).add(new Edge(j, 0.0, 0, 0));  // Assuming 0, 0 for x, y coordinates
                    graph.get(j).add(new Edge(i, 0.0, 0, 0));  // Assuming 0, 0 for x, y coordinates
                } else {
                    // Neither have drones (use actual distance)
                    graph.get(i).add(new Edge(j, distance, graph.get(j).get(0).x, graph.get(j).get(0).y));
                    graph.get(j).add(new Edge(i, distance, graph.get(i).get(0).x, graph.get(i).get(0).y));
                }
            }
        }

        // Minimum Spanning Tree using Prim's algorithm
        double minThreshold = primMST(graph);

        // Print minimum threshold for this test case
        System.out.printf("%.2f\n", minThreshold);
    }

    sc.close();
  }

    static class Edge {
    int to;
    double weight;
    int x;
    int y;

    public Edge(int to, double weight, int x, int y) {
        this.to = to;
        this.weight = weight;
        this.x = x;
        this.y = y;
    }
}



static double primMST(Map<Integer, List<Edge>> graph) {
    // Priority queue to store edges sorted by weight (ascending order)
    PriorityQueue<Edge> pq = new PriorityQueue<>((e1, e2) -> Double.compare(e1.weight, e2.weight));

    // Set to keep track of visited vertices
    Set<Integer> visited = new HashSet<>();

    // Choose any starting vertex (here, we choose vertex 1)
    int start = 1;
    pq.addAll(graph.get(start));
    visited.add(start);

    double totalWeight = 0.0; // To store total weight of MST edges

    // Prim's algorithm loop
    while (!pq.isEmpty()) {
        Edge currentEdge = pq.poll();
        int to = currentEdge.to;

        // Skip already visited vertices
        if (visited.contains(to)) {
        continue;
        }

        // Update total weight
        totalWeight += currentEdge.weight;

        // Add edges from the newly visited vertex to the priority queue
        visited.add(to);
        pq.addAll(graph.get(to));
    }

    // Return the total weight of the MST
    return totalWeight;
    }
