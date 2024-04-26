import java.util.ArrayList;


public class PowerGridOptimization {
    private ArrayList<Integer> amountOfEnergyDemandsArrivingPerHour;

    public PowerGridOptimization(ArrayList<Integer> energyDemands){
        this.amountOfEnergyDemandsArrivingPerHour = energyDemands;
    }

    public OptimalPowerGridSolution getOptimalPowerGridSolutionDP(){
        int N = amountOfEnergyDemandsArrivingPerHour.size();
        int[] SOL = new int[N+1]; // To store the max satisfied GWs till hour j
        ArrayList<ArrayList<Integer>> HOURS = new ArrayList<>(); // To store the discharge hours leading to SOL[j]

        // Initialize SOL and HOURS for hour 0
        SOL[0] = 0; // No energy can be satisfied at hour 0
        HOURS.add(new ArrayList<>()); // No discharge hours at hour 0

        // Dynamic programming to fill SOL and HOURS
        for (int j = 1; j <= N; j++) {
            SOL[j] = 0; // Initialize SOL[j]
            ArrayList<Integer> optimalHours = new ArrayList<>();
            for (int i = 0; i < j; i++) {
                int currentSatisfaction = SOL[i] + Math.min(amountOfEnergyDemandsArrivingPerHour.get(j-1), (int)Math.pow(j-i, 2));
                if (currentSatisfaction > SOL[j]) {
                    SOL[j] = currentSatisfaction;
                    optimalHours = new ArrayList<>(HOURS.get(i)); // Copy the optimal hours from HOURS[i]
                    if (!optimalHours.contains(j)) {
                        optimalHours.add(j); // Add current hour to the discharge plan
                    }
                }
            }
            HOURS.add(optimalHours);
        }

        // The last entry in SOL contains the maximum number of GWs that can be satisfied
        // The last entry in HOURS contains the discharge hours for the optimal solution
        int maxSatisfiedGWs = SOL[N];
        ArrayList<Integer> dischargeHours = HOURS.get(N);
        int totalDemandedGWs = amountOfEnergyDemandsArrivingPerHour.stream().mapToInt(Integer::intValue).sum();
        int unsatisfiedGWs = totalDemandedGWs - maxSatisfiedGWs;

        // Constructing and returning the optimal solution
        return new OptimalPowerGridSolution(maxSatisfiedGWs, dischargeHours, unsatisfiedGWs, totalDemandedGWs);
    }
}
