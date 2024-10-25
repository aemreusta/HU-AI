import java.util.ArrayList;
import java.util.Collections;

public class OptimalESVDeploymentGP {
    private ArrayList<Integer> maintenanceTaskEnergyDemands;
    private ArrayList<ArrayList<Integer>> maintenanceTasksAssignedToESVs = new ArrayList<>();

    public OptimalESVDeploymentGP(ArrayList<Integer> maintenanceTaskEnergyDemands) {
        this.maintenanceTaskEnergyDemands = new ArrayList<>(maintenanceTaskEnergyDemands);
    }    

    public int getMinNumESVsToDeploy(int maxNumberOfAvailableESVs, int maxESVCapacity) {
        // Sort the tasks in decreasing order of energy demand
        Collections.sort(maintenanceTaskEnergyDemands, Collections.reverseOrder());

        // Initialize ESVs as empty bins with maximum capacity
        int[] esvCapacities = new int[maxNumberOfAvailableESVs];
        for (int i = 0; i < maxNumberOfAvailableESVs; i++) {
            esvCapacities[i] = maxESVCapacity; // Each ESV starts with max capacity
            maintenanceTasksAssignedToESVs.add(new ArrayList<>()); // Initialize the list for assigned tasks
        }

        // Try to fit each task in the first ESV where it fits
        for (int task : maintenanceTaskEnergyDemands) {
            boolean fit = false;
            for (int i = 0; i < maxNumberOfAvailableESVs && !fit; i++) {
                if (esvCapacities[i] >= task) {
                    esvCapacities[i] -= task; // Decrease the available capacity
                    maintenanceTasksAssignedToESVs.get(i).add(task); // Assign task to this ESV
                    fit = true;
                }
            }
            if (!fit) {
                return -1; // Task couldn't be fit in any ESV
            }
        }

        // Count non-empty ESVs to find out how many were actually used
        int usedESVs = 0;
        for (ArrayList<Integer> esvTasks : maintenanceTasksAssignedToESVs) {
            if (!esvTasks.isEmpty()) {
                usedESVs++;
            }
        }

        // Prune empty lists to reflect actual assignments
        maintenanceTasksAssignedToESVs.removeIf(ArrayList::isEmpty);

        return usedESVs;
    }

    public ArrayList<ArrayList<Integer>> getMaintenanceTasksAssignedToESVs() {
        return maintenanceTasksAssignedToESVs;
    }
}
