import java.io.File;
import java.util.ArrayList;
import java.util.List;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

public class MissionExploration {

    /**
     * Given a Galaxy object, prints the solar systems within that galaxy.
     * Uses exploreSolarSystems() and printSolarSystems() methods of the Galaxy object.
     *
     * @param galaxy a Galaxy object
     */
    public void printSolarSystems(Galaxy galaxy) {
        List<SolarSystem> solarSystems = galaxy.exploreSolarSystems();
        galaxy.printSolarSystems(solarSystems);
    }

    /**
     * Parse the input XML file and return a Galaxy object populated with Planet instances.
     *
     * @param filename the input XML file
     * @return a Galaxy object containing parsed planets
     */
    public Galaxy readXML(String filename) {
        List<Planet> planetList = new ArrayList<>();
    
        try {
            File inputFile = new File(filename);
            DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
            Document doc = dBuilder.parse(inputFile);
            doc.getDocumentElement().normalize();
    
            NodeList planetNodes = doc.getElementsByTagName("Planet");
    
            for (int temp = 0; temp < planetNodes.getLength(); temp++) {
                Node planetNode = planetNodes.item(temp);
                if (planetNode.getNodeType() == Node.ELEMENT_NODE) {
                    Element planetElement = (Element) planetNode;
                    
                    String id = planetElement.getElementsByTagName("ID").item(0).getTextContent().trim();
                    int blackHoleProximity = Integer.parseInt(planetElement.getElementsByTagName("BlackHoleProximity").item(0).getTextContent().trim());
                    
                    NodeList neighborsNodeList = planetElement.getElementsByTagName("Neighbors").item(0).getChildNodes();
                    List<String> neighbors = new ArrayList<>();
                    
                    for (int i = 0; i < neighborsNodeList.getLength(); i++) {
                        Node neighborNode = neighborsNodeList.item(i);
                        if (neighborNode.getNodeType() == Node.ELEMENT_NODE) {
                            Element neighborElement = (Element) neighborNode;
                            neighbors.add(neighborElement.getTextContent().trim());
                        }
                    }
                    
                    boolean hasDuplicator = planetElement.getElementsByTagName("Duplicator").getLength() > 0;
                    
                    // If the planet has a duplicator, create two planets with the same ID and neighbors but adding a "C" to the end of the ID
                    if (hasDuplicator) {
                        Planet planet = new Planet(id, blackHoleProximity, neighbors);
                        planetList.add(planet);
                        Planet duplicator = new Planet(id + "C", blackHoleProximity, neighbors);
                        planetList.add(duplicator);
                    }
                    
                    else {
                        // Create the Planet object
                        Planet planet = new Planet(id, blackHoleProximity, neighbors);
                        planetList.add(planet);
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    
        return new Galaxy(planetList);
    }
    
}
