import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class MissionGenesis {

    private MolecularData molecularDataHuman; // Holds molecular data for humans
    private MolecularData molecularDataVitales; // Holds molecular data for a species or group called Vitales

    // Getter for human molecular data
    public MolecularData getMolecularDataHuman() {
        return molecularDataHuman;
    }

    // Getter for Vitales molecular data
    public MolecularData getMolecularDataVitales() {
        return molecularDataVitales;
    }

    // Method to read and process XML data file
    public void readXML(String filename) {
        try {
            File inputFile = new File(filename); // Create a File instance with the given filename
            DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance(); // Get DocumentBuilderFactory instance
            DocumentBuilder dBuilder = dbFactory.newDocumentBuilder(); // Create a new DocumentBuilder
            Document doc = dBuilder.parse(inputFile); // Parse the input file into a Document
            doc.getDocumentElement().normalize(); // Normalize the document structure

            // Retrieve elements by tag name for both Human and Vitales molecular data
            NodeList humanList = doc.getElementsByTagName("HumanMolecularData");
            NodeList vitalesList = doc.getElementsByTagName("VitalesMolecularData");

            // Process the NodeList for Human Molecular Data
            List<Molecule> humanMolecules = processMolecularData(humanList);
            molecularDataHuman = new MolecularData(humanMolecules); // Initialize molecular data for humans

            // Process the NodeList for Vitales Molecular Data
            List<Molecule> vitalesMolecules = processMolecularData(vitalesList);
            molecularDataVitales = new MolecularData(vitalesMolecules); // Initialize molecular data for Vitales

        } catch (Exception e) {
            e.printStackTrace(); // Print any errors during XML processing
        }
    }

    // Method to process NodeList and extract molecular data
    private List<Molecule> processMolecularData(NodeList nodeList) {
        List<Molecule> molecules = new ArrayList<>();
        for (int i = 0; i < nodeList.getLength(); i++) {
            Node node = nodeList.item(i);
            if (node.getNodeType() == Node.ELEMENT_NODE) { // Check if the node is an element node
                Element element = (Element) node;
                NodeList moleculeList = element.getElementsByTagName("Molecule");
                for (int j = 0; j < moleculeList.getLength(); j++) {
                    Node moleculeNode = moleculeList.item(j);
                    if (moleculeNode.getNodeType() == Node.ELEMENT_NODE) {
                        Element moleculeElement = (Element) moleculeNode;
                        String id = moleculeElement.getElementsByTagName("ID").item(0).getTextContent(); // Get molecule ID
                        int bondStrength = Integer.parseInt(moleculeElement.getElementsByTagName("BondStrength").item(0).getTextContent()); // Get bond strength
                        NodeList bondsNodeList = moleculeElement.getElementsByTagName("MoleculeID"); // Get bonded molecule IDs
                        List<String> bonds = new ArrayList<>();
                        for (int k = 0; k < bondsNodeList.getLength(); k++) {
                            bonds.add(bondsNodeList.item(k).getTextContent()); // Collect all bond IDs
                        }
                        Molecule molecule = new Molecule(id, bondStrength, bonds); // Create Molecule object
                        molecules.add(molecule); // Add molecule to list
                    }
                }
            }
        }
        return molecules; // Return list of molecules
    }
}
