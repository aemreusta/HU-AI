import org.w3c.dom.*;
import javax.xml.parsers.*;
import java.io.File;
import java.util.*;

public class AlienFlora {
    private File xmlFile;

    public AlienFlora(File xmlFile) {
        this.xmlFile = xmlFile;
    }

    public void readGenomes() {
        // TODO:
        // - Parse XML
        // - Read genomes and links
        // - Create clusters
        // - Print number of clusters and their genome IDs
    }

    public void evaluateEvolutions() {
        // TODO:
        // - Parse and process possibleEvolutionPairs
        // - Find min evolution genome in each cluster
        // - Calculate and print evolution factors
    }

    public void evaluateAdaptations() {
        // TODO:
        // - Parse and process possibleAdaptationPairs
        // - If genomes in same cluster, use Dijkstra to calculate min path
        // - Print adaptation factors
    }
}
