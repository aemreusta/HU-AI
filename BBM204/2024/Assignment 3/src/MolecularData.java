import java.util.*;
import java.util.stream.Collectors;

public class MolecularData {

    private final List<Molecule> molecules;

    public MolecularData(List<Molecule> molecules) {
        this.molecules = new ArrayList<>(molecules); // make a defensive copy if necessary
    }


    public List<Molecule> getMolecules() {
        return molecules;
    }

    public void addMolecule(Molecule molecule) {
        this.molecules.add(molecule);
    }

    public List<MolecularStructure> identifyMolecularStructures() {
        Map<String, MolecularStructure> moleculeToStructure = new HashMap<>();
        List<MolecularStructure> structures = new ArrayList<>(); // Temporary list to hold all structures

        for (Molecule molecule : molecules) {
            if (!moleculeToStructure.containsKey(molecule.getId())) {
                MolecularStructure newStructure = new MolecularStructure();
                dfs(molecule, newStructure, moleculeToStructure);
                if (!newStructure.getMolecules().isEmpty()) {
                    structures.add(newStructure); // Only add non-empty structures
                }
            }
        }

        // Clean up the list to remove any empty structures if any are left
        return structures.stream()
                .filter(structure -> !structure.getMolecules().isEmpty())
                .collect(Collectors.toList());
    }

    public void mergeStructures(MolecularStructure structure1, MolecularStructure structure2, Map<String, MolecularStructure> moleculeToStructure) {
        if (structure1 == structure2) {
            // If both are the same structure, no need to merge
            return;
        }

        // Merge molecules from structure2 into structure1
        for (Molecule molecule : structure2.getMolecules()) {
            structure1.addMolecule(molecule);
            moleculeToStructure.put(molecule.getId(), structure1);  // Update the mapping
        }

        // Optional: Clear the merged structure to release resources, if necessary
        structure2.getMolecules().clear();
    }

    private void dfs(Molecule molecule, MolecularStructure currentStructure, Map<String, MolecularStructure> moleculeToStructure) {
        if (moleculeToStructure.containsKey(molecule.getId())) {
            MolecularStructure existingStructure = moleculeToStructure.get(molecule.getId());
            if (existingStructure != currentStructure) {
                mergeStructures(currentStructure, existingStructure, moleculeToStructure);
            }
            return;
        }

        moleculeToStructure.put(molecule.getId(), currentStructure);
        currentStructure.addMolecule(molecule);

        for (String bondedId : molecule.getBonds()) {
            Molecule bondedMolecule = findMoleculeById(bondedId);
            if (bondedMolecule != null) {
                dfs(bondedMolecule, currentStructure, moleculeToStructure);
            }
        }
    }


    private Molecule findMoleculeById(String id) {
        for (Molecule m : molecules) {
            if (m.getId().equals(id)) {
                return m;
            }
        }
        return null; // Consider throwing an exception or handling this case properly.
    }



    public void printMolecularStructures(List<MolecularStructure> molecularStructures, String species) {
        System.out.println(molecularStructures.size() + " molecular structures have been discovered in " + species + ".");
        int count = 1;
        for (MolecularStructure structure : molecularStructures) {
            System.out.println("Molecules in Molecular Structure " + count + ": " + structure);
            count++;
        }
    }

    public void printVitalesAnomaly(List<MolecularStructure> sourceStructures, List<MolecularStructure> targetStructures) {
        List<MolecularStructure> uniqueStructures = getVitalesAnomaly(sourceStructures, targetStructures);
        System.out.println("Molecular structures unique to Vitales individuals:");
        for (MolecularStructure unique : uniqueStructures) {
            System.out.println(unique);
        }
    }

    public static ArrayList<MolecularStructure> getVitalesAnomaly(List<MolecularStructure> sourceStructures, List<MolecularStructure> targetStructures) {
        Set<String> sourceSignatures = sourceStructures.stream()
                .map(MolecularStructure::toString)
                .collect(Collectors.toSet());
        return targetStructures.stream()
                .filter(structure -> !sourceSignatures.contains(structure.toString()))
                .collect(Collectors.toCollection(ArrayList::new));  // Collect results into an ArrayList
    }


}
