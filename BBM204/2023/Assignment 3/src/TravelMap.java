import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import java.io.File;
import java.util.*;

public class TravelMap {

    // Maps a single Id to a single Location.
    public Map<Integer, Location> locationMap = new HashMap<>();

    // List of locations, read in the given order
    public List<Location> locations = new ArrayList<>();

    // List of trails, read in the given order
    public List<Trail> trails = new ArrayList<>();

    // TODO: You are free to add more variables if necessary.

    public void initializeMap(String filename) {
        try {
            File inputFile = new File(filename);

            // Create a document builder factory object
            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            DocumentBuilder builder = factory.newDocumentBuilder();

            // Parse the XML file
            Document doc = builder.parse(inputFile);

            // Get the Locations and Trails elements from the XML file
            Element locationsElement = (Element) doc.getElementsByTagName("Locations").item(0);
            Element trailsElement = (Element) doc.getElementsByTagName("Trails").item(0);

            // Read in the locations
            NodeList locationNodes = locationsElement.getElementsByTagName("Location");
            for (int i = 0; i < locationNodes.getLength(); i++) {
                Element locationElement = (Element) locationNodes.item(i);
                int id = Integer.parseInt(locationElement.getElementsByTagName("Id").item(0).getTextContent());
                String name = locationElement.getElementsByTagName("Name").item(0).getTextContent();
                Location location = new Location(name, id);
                locationMap.put(id, location);
                locations.add(location);
            }

            // Read in the trails
            NodeList trailNodes = trailsElement.getElementsByTagName("Trail");
            for (int i = 0; i < trailNodes.getLength(); i++) {
                Element trailElement = (Element) trailNodes.item(i);
                int sourceId = Integer.parseInt(trailElement.getElementsByTagName("Source").item(0).getTextContent());
                int destinationId = Integer
                        .parseInt(trailElement.getElementsByTagName("Destination").item(0).getTextContent());
                int danger = Integer.parseInt(trailElement.getElementsByTagName("Danger").item(0).getTextContent());
                Trail trail = new Trail(locationMap.get(sourceId), locationMap.get(destinationId), danger);
                trails.add(trail);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public List<Trail> getSafestTrails() {
        List<Trail> safestTrails = new ArrayList<>();
        // Fill the safestTrail list and return it.
        // Select the optimal Trails from the Trail list that you have read.
        // TODO: Your code here
        return safestTrails;
    }

    public void printSafestTrails(List<Trail> safestTrails) {
        // Print the given list of safest trails conforming to the given output format.
        // TODO: Your code here
    }
}
