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

    private MolecularData molecularDataHuman; // Molecular data for humans
    private MolecularData molecularDataVitales; // Molecular data for Vitales

    public MolecularData getMolecularDataHuman() {
        return molecularDataHuman;
    }

    public MolecularData getMolecularDataVitales() {
        return molecularDataVitales;
    }

    public void readXML(String filename) {
        try {
            File inputFile = new File(filename);
            DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
            Document doc = dBuilder.parse(inputFile);
            doc.getDocumentElement().normalize();

            // Assume XML structure includes separate child elements for Human and Vitales data
            NodeList humanList = doc.getElementsByTagName("HumanMolecularData");
            NodeList vitalesList = doc.getElementsByTagName("VitalesMolecularData");

            // Process Human Molecular Data
            List<Molecule> humanMolecules = processMolecularData(humanList);
            molecularDataHuman = new MolecularData(humanMolecules);

            // Process Vitales Molecular Data
            List<Molecule> vitalesMolecules = processMolecularData(vitalesList);
            molecularDataVitales = new MolecularData(vitalesMolecules);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private List<Molecule> processMolecularData(NodeList nodeList) {
        List<Molecule> molecules = new ArrayList<>();
        for (int i = 0; i < nodeList.getLength(); i++) {
            Node node = nodeList.item(i);
            if (node.getNodeType() == Node.ELEMENT_NODE) {
                Element element = (Element) node;
                NodeList moleculeList = element.getElementsByTagName("Molecule");
                for (int j = 0; j < moleculeList.getLength(); j++) {
                    Node moleculeNode = moleculeList.item(j);
                    if (moleculeNode.getNodeType() == Node.ELEMENT_NODE) {
                        Element moleculeElement = (Element) moleculeNode;
                        String id = moleculeElement.getElementsByTagName("ID").item(0).getTextContent();
                        int bondStrength = Integer.parseInt(moleculeElement.getElementsByTagName("BondStrength").item(0).getTextContent());
                        NodeList bondsNodeList = moleculeElement.getElementsByTagName("MoleculeID");
                        List<String> bonds = new ArrayList<>();
                        for (int k = 0; k < bondsNodeList.getLength(); k++) {
                            bonds.add(bondsNodeList.item(k).getTextContent());
                        }
                        Molecule molecule = new Molecule(id, bondStrength, bonds);
                        molecules.add(molecule);
                    }
                }
            }
        }
        return molecules;
    }
}
