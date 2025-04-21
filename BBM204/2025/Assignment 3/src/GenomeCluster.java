import java.util.*;

public class GenomeCluster {
    public Map<String, Genome> genomeMap = new HashMap<>();

    public void addGenome(Genome genome) {
        genomeMap.put(genome.id, genome);
    }

    public boolean contains(String genomeId) {
        return genomeMap.containsKey(genomeId);
    }

    public Genome getMinEvolutionGenome() {
        return genomeMap.values().stream()
                .min(Comparator.comparingInt(g -> g.evolutionFactor))
                .orElse(null);
    }

    public int dijkstra(String startId, String endId) {
        if (!genomeMap.containsKey(startId) || !genomeMap.containsKey(endId)) {
            return -1;
        }

        Map<String, List<Edge>> adjList = new HashMap<>();
        for (Genome genome : genomeMap.values()) {
            List<Edge> edges = new ArrayList<>();
            for (Genome.Link link : genome.links) {
                if (genomeMap.containsKey(link.target)) {
                    edges.add(new Edge(link.target, link.adaptationFactor));
                }
            }
            adjList.put(genome.id, edges);
        }

        PriorityQueue<Node> pq = new PriorityQueue<>();
        Map<String, Integer> dist = new HashMap<>();
        for (String id : genomeMap.keySet()) {
            dist.put(id, Integer.MAX_VALUE);
        }
        dist.put(startId, 0);
        pq.add(new Node(startId, 0));

        while (!pq.isEmpty()) {
            Node current = pq.poll();
            String u = current.id;
            int currentDist = current.distance;

            if (u.equals(endId)) {
                return currentDist;
            }

            if (currentDist > dist.get(u)) {
                continue;
            }

            for (Edge edge : adjList.getOrDefault(u, Collections.emptyList())) {
                String v = edge.target;
                int weight = edge.weight;
                int newDist = currentDist + weight;
                if (newDist < dist.get(v)) {
                    dist.put(v, newDist);
                    pq.add(new Node(v, newDist));
                }
            }
        }

        return -1;
    }

    private static class Edge {
        String target;
        int weight;

        Edge(String target, int weight) {
            this.target = target;
            this.weight = weight;
        }
    }

    private static class Node implements Comparable<Node> {
        String id;
        int distance;

        Node(String id, int distance) {
            this.id = id;
            this.distance = distance;
        }

        @Override
        public int compareTo(Node other) {
            return Integer.compare(this.distance, other.distance);
        }
    }
}