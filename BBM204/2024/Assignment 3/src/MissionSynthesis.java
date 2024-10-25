import java.util.*;

public class MissionSynthesis {

    // List to hold human-derived molecular structures
    private final List<MolecularStructure> humanStructures;
    // List to hold molecular structures from other sources
    private final ArrayList<MolecularStructure> diffStructures;

    // Constructor to initialize humanStructures and diffStructures
    public MissionSynthesis(List<MolecularStructure> humanStructures, ArrayList<MolecularStructure> diffStructures) {
        this.humanStructures = humanStructures;
        this.diffStructures = diffStructures;
    }

    // Method to synthesize a serum by forming bonds between molecules
    public List<Bond> synthesizeSerum() {
        List<Bond> serum = new ArrayList<>(); // List to store the resulting bonds forming the serum
        List<Bond> potentialBonds = new ArrayList<>(); // Temporary list to store all possible bonds

        // Collect all molecules with the lowest bond strength from each structure
        for (MolecularStructure ms : humanStructures) {
            if (!ms.getMolecules().isEmpty()) {
                Molecule minBondMolecule = Collections.min(ms.getMolecules(), Comparator.comparingInt(Molecule::getBondStrength));
                for (MolecularStructure ms2 : diffStructures) {
                    if (!ms2.getMolecules().isEmpty()) {
                        Molecule minBondMolecule2 = Collections.min(ms2.getMolecules(), Comparator.comparingInt(Molecule::getBondStrength));
                        double avgBondStrength = (minBondMolecule.getBondStrength() + minBondMolecule2.getBondStrength()) / 2.0;
                        potentialBonds.add(new Bond(minBondMolecule, minBondMolecule2, avgBondStrength));
                    }
                }
            }
        }

        // Sort bonds by weight (bond strength)
        Collections.sort(potentialBonds, Comparator.comparingDouble(Bond::getWeight));

        // Implementing Kruskal's algorithm to select optimal bonds
        Map<Molecule, Molecule> parent = new HashMap<>();
        for (MolecularStructure ms : humanStructures) {
            for (Molecule m : ms.getMolecules()) {
                parent.put(m, m); // Each molecule is its own parent initially
            }
        }
        for (MolecularStructure ms : diffStructures) {
            for (Molecule m : ms.getMolecules()) {
                parent.put(m, m); // Each molecule is its own parent initially
            }
        }

        for (Bond bond : potentialBonds) {
            Molecule root1 = find(parent, bond.getFrom());
            Molecule root2 = find(parent, bond.getTo());
            if (!root1.equals(root2)) {
                serum.add(bond);
                parent.put(root1, root2); // Union operation to form a spanning tree
            }
        }

        return serum;
    }

    // Method to find the root of a molecule, with path compression for efficiency
    private Molecule find(Map<Molecule, Molecule> parent, Molecule m) {
        if (parent.get(m) != m) {
            parent.put(m, find(parent, parent.get(m))); // Path compression
        }
        return parent.get(m);
    }

    // Method to print the details of the synthesized serum
    public void printSynthesis(List<Bond> serum) {
        for (Bond bond : serum) {
            // Ensuring the molecules are printed in ascending order by their ID
            Molecule first = bond.getFrom();
            Molecule second = bond.getTo();
    
            // Compare IDs to determine order
            if (first.getId().compareTo(second.getId()) > 0) {
                // Swap if 'first' should actually come second
                Molecule temp = first;
                first = second;
                second = temp;
            }
    
            // Formatting the output string
            System.out.println("Forming a bond between " + first.getId() + " - " + second.getId() + " with strength " + String.format("%.2f", bond.getWeight()));
        }
    
        // Optionally, if you want to print the total bond strength of the serum, you can add:
        double totalStrength = serum.stream().mapToDouble(Bond::getWeight).sum();
        System.out.println("The total serum bond strength is " + String.format("%.2f", totalStrength));
    }    
}
