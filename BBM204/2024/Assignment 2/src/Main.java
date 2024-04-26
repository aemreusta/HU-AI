import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Main class
 */
public class Main {
    public static void main(String[] args) throws IOException {
        /** MISSION POWER GRID OPTIMIZATION BELOW **/

        System.out.println("##MISSION POWER GRID OPTIMIZATION##");
        // Reading the demand schedule from the file
        String demandScheduleFilePath = args[0]; // Getting the first command-line argument
        List<Integer> energyDemands = new ArrayList<>();
        Files.lines(Paths.get(demandScheduleFilePath)).forEach(line ->
                Arrays.stream(line.split(" ")).forEach(number ->
                        energyDemands.add(Integer.parseInt(number))));

        // Instantiating the PowerGridOptimization object and calling the method
        PowerGridOptimization optimization = new PowerGridOptimization(energyDemands);
        OptimalPowerGridSolution optimalSolution = optimization.getOptimalPowerGridSolutionDP();

        // Printing the solution to STDOUT
        System.out.println("The total number of demanded gigawatts: " + optimalSolution.getTotalDemandedGigawatts());
        System.out.println("Maximum number of satisfied gigawatts: " + optimalSolution.getMaxNumberOfSatisfiedDemands());
        System.out.println("Hours at which the battery bank should be discharged: " + optimalSolution.getHoursToDischargeBatteriesForMaxEfficiency());
        System.out.println("The number of unsatisfied gigawatts: " + optimalSolution.getUnsatisfiedGigawatts());
        System.out.println("##MISSION POWER GRID OPTIMIZATION COMPLETED##");

        /** MISSION ECO-MAINTENANCE BELOW **/

        System.out.println("##MISSION ECO-MAINTENANCE##");
        // Reading the ESV maintenance data from the file
        String esvMaintenanceFilePath = args[1]; // Getting the second command-line argument
        List<String> lines = Files.readAllLines(Paths.get(esvMaintenanceFilePath));
        String[] firstLineSplit = lines.get(0).split(" ");
        int maxNumberOfAvailableESVs = Integer.parseInt(firstLineSplit[0]);
        int esvCapacity = Integer.parseInt(firstLineSplit[1]);
        List<Integer> maintenanceTasks = Arrays.stream(lines.get(1).split(" "))
                .map(Integer::parseInt)
                .toList();

        // Instantiating the OptimalESVDeploymentGP object and calling the method
        OptimalESVDeploymentGP esvDeployment = new OptimalESVDeploymentGP(maintenanceTasks);
        Map<Integer, List<Integer>> esvAssignments = esvDeployment.getMinNumESVsToDeploy(maxNumberOfAvailableESVs, esvCapacity);

        // Checking if all tasks could be accommodated and printing the solution to STDOUT
        if (esvAssignments == null) {
            System.out.println("Warning: Mission Eco-Maintenance Failed.");
        } else {
            System.out.println("The minimum number of ESVs to deploy: " + esvAssignments.size());
            esvAssignments.forEach((esv, tasks) -> {
                System.out.println("ESV " + esv + " tasks: " + tasks);
            });
        }
        System.out.println("##MISSION ECO-MAINTENANCE COMPLETED##");
    }
}
