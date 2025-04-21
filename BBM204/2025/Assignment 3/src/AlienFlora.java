import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import java.io.File;
import java.util.*;

public class AlienFlora {
    private File xmlFile;
    private List<GenomeCluster> clusters = new ArrayList<>();
    private Map<String, Genome> genomeMap = new LinkedHashMap<>();

    public AlienFlora(File xmlFile) {
        this.xmlFile = xmlFile;
    }

    public void readGenomes() {
        try {
            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            DocumentBuilder builder = factory.newDocumentBuilder();
            Document doc = builder.parse(xmlFile);
            doc.getDocumentElement().normalize();

            Map<String, Set<String>> undirectedAdjList = new HashMap<>();

            NodeList genomeNodes = doc.getElementsByTagName("genome");
            for (int i = 0; i < genomeNodes.getLength(); i++) {
                Element genomeElement = (Element) genomeNodes.item(i);
                String id = genomeElement.getElementsByTagName("id").item(0).getTextContent().trim();
                int evolutionFactor = Integer.parseInt(genomeElement.getElementsByTagName("evolutionFactor").item(0).getTextContent().trim());
                Genome genome = new Genome(id, evolutionFactor);
                genomeMap.put(id, genome);

                NodeList linkNodes = genomeElement.getElementsByTagName("link");
                for (int j = 0; j < linkNodes.getLength(); j++) {
                    Element linkElement = (Element) linkNodes.item(j);
                    String target = linkElement.getElementsByTagName("target").item(0).getTextContent().trim();
                    int adaptationFactor = Integer.parseInt(linkElement.getElementsByTagName("adaptationFactor").item(0).getTextContent().trim());
                    genome.addLink(target, adaptationFactor);

                    undirectedAdjList.computeIfAbsent(id, k -> new LinkedHashSet<>()).add(target);
                    undirectedAdjList.computeIfAbsent(target, k -> new LinkedHashSet<>()).add(id);
                }
            }

            Set<String> visited = new HashSet<>();
            for (String id : genomeMap.keySet()) {
                if (!visited.contains(id)) {
                    Queue<String> queue = new LinkedList<>();
                    Set<String> clusterIds = new LinkedHashSet<>();
                    queue.add(id);
                    visited.add(id);
                    clusterIds.add(id);

                    while (!queue.isEmpty()) {
                        String current = queue.poll();
                        for (String neighbor : undirectedAdjList.getOrDefault(current, Collections.emptySet())) {
                            if (!visited.contains(neighbor)) {
                                visited.add(neighbor);
                                queue.add(neighbor);
                                clusterIds.add(neighbor);
                            }
                        }
                    }

                    GenomeCluster cluster = new GenomeCluster();
                    for (String genomeId : clusterIds) {
                        cluster.addGenome(genomeMap.get(genomeId));
                    }
                    clusters.add(cluster);
                }
            }

            System.out.println("##Start Reading Flora Genomes##");
            System.out.println("Number of Genome Clusters: " + clusters.size());
            List<List<String>> clusterOutput = new ArrayList<>();
            for (GenomeCluster cluster : clusters) {
                List<String> ids = new ArrayList<>(cluster.genomeMap.keySet());
                clusterOutput.add(ids);
            }
            System.out.println("For the Genomes: " + clusterOutput);
            System.out.println("##Reading Flora Genomes Completed##");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void evaluateEvolutions() {
        try {
            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            DocumentBuilder builder = factory.newDocumentBuilder();
            Document doc = builder.parse(xmlFile);
            doc.getDocumentElement().normalize();

            List<String[]> pairs = new ArrayList<>();
            NodeList pairNodes = doc.getElementsByTagName("possibleEvolutionPairs");
            if (pairNodes.getLength() > 0) {
                Element pairsElement = (Element) pairNodes.item(0);
                NodeList pairList = pairsElement.getElementsByTagName("pair");
                for (int i = 0; i < pairList.getLength(); i++) {
                    Element pairElement = (Element) pairList.item(i);
                    String firstId = pairElement.getElementsByTagName("firstId").item(0).getTextContent().trim();
                    String secondId = pairElement.getElementsByTagName("secondId").item(0).getTextContent().trim();
                    pairs.add(new String[]{firstId, secondId});
                }
            }

            System.out.println("##Start Evaluating Possible Evolutions##");
            System.out.println("Number of Possible Evolutions: " + pairs.size());
            List<Double> evolutionFactors = new ArrayList<>();
            int certifiedCount = 0;

            for (String[] pair : pairs) {
                String firstId = pair[0];
                String secondId = pair[1];
                GenomeCluster firstCluster = null, secondCluster = null;

                for (GenomeCluster cluster : clusters) {
                    if (cluster.contains(firstId)) {
                        firstCluster = cluster;
                    }
                    if (cluster.contains(secondId)) {
                        secondCluster = cluster;
                    }
                }

                if (firstCluster == null || secondCluster == null || firstCluster == secondCluster) {
                    evolutionFactors.add(-1.0);
                } else {
                    Genome minFirst = firstCluster.getMinEvolutionGenome();
                    Genome minSecond = secondCluster.getMinEvolutionGenome();
                    double avg = (minFirst.evolutionFactor + minSecond.evolutionFactor) / 2.0;
                    evolutionFactors.add(avg);
                    certifiedCount++;
                }
            }

            System.out.println("Number of Certified Evolution: " + certifiedCount);
            System.out.println("Evolution Factor for Each Evolution Pair: " + evolutionFactors);
            System.out.println("##Evaluated Possible Evolutions##");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void evaluateAdaptations() {
        try {
            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            DocumentBuilder builder = factory.newDocumentBuilder();
            Document doc = builder.parse(xmlFile);
            doc.getDocumentElement().normalize();

            List<String[]> pairs = new ArrayList<>();
            NodeList pairNodes = doc.getElementsByTagName("possibleAdaptationPairs");
            if (pairNodes.getLength() > 0) {
                Element pairsElement = (Element) pairNodes.item(0);
                NodeList pairList = pairsElement.getElementsByTagName("pair");
                for (int i = 0; i < pairList.getLength(); i++) {
                    Element pairElement = (Element) pairList.item(i);
                    String firstId = pairElement.getElementsByTagName("firstId").item(0).getTextContent().trim();
                    String secondId = pairElement.getElementsByTagName("secondId").item(0).getTextContent().trim();
                    pairs.add(new String[]{firstId, secondId});
                }
            }

            System.out.println("##Start Evaluating Possible Adaptations##");
            System.out.println("Number of Possible Adaptations: " + pairs.size());
            List<Integer> adaptationFactors = new ArrayList<>();
            int certifiedCount = 0;

            for (String[] pair : pairs) {
                String firstId = pair[0];
                String secondId = pair[1];
                GenomeCluster cluster = null;

                for (GenomeCluster c : clusters) {
                    if (c.contains(firstId) && c.contains(secondId)) {
                        cluster = c;
                        break;
                    }
                }

                if (cluster == null) {
                    adaptationFactors.add(-1);
                } else {
                    int factor = cluster.dijkstra(firstId, secondId);
                    adaptationFactors.add(factor == -1 ? -1 : factor);
                    if (factor != -1) {
                        certifiedCount++;
                    }
                }
            }

            System.out.println("Number of Certified Adaptations: " + certifiedCount);
            System.out.println("Adaptation Factor for Each Adaptation Pair: " + adaptationFactors);
            System.out.println("##Evaluated Possible Evolutions##");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}