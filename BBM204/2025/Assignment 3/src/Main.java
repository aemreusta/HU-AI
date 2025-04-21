import java.io.File;

public class Main {
    public static void main(String[] args) {

        // File xmlFile = new File(args[0]);
        File xmlFile = new File("/Users/emre/GitHub/HU-AI/BBM204/2025/Assignment 3/src/AlienFlora.xml");
        AlienFlora alienFlora = new AlienFlora(xmlFile);
        alienFlora.readGenomes();
        alienFlora.evaluateEvolutions();
        alienFlora.evaluateAdaptations();
    }
}