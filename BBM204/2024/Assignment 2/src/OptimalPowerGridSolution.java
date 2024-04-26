import java.util.ArrayList;

public class OptimalPowerGridSolution {
    private int maxNumberOfSatisfiedDemands;
    private ArrayList<Integer> hoursToDischargeBatteriesForMaxEfficiency;
    private int totalDemandedGigawatts;
    private int unsatisfiedGigawatts;

    // Constructor with all the necessary parameters
    public OptimalPowerGridSolution(int maxNumberOfSatisfiedDemands, ArrayList<Integer> hoursToDischargeBatteriesForMaxEfficiency, int unsatisfiedGigawatts, int totalDemandedGigawatts) {
        this.maxNumberOfSatisfiedDemands = maxNumberOfSatisfiedDemands;
        this.hoursToDischargeBatteriesForMaxEfficiency = hoursToDischargeBatteriesForMaxEfficiency;
        this.unsatisfiedGigawatts = unsatisfiedGigawatts;
        this.totalDemandedGigawatts = totalDemandedGigawatts;
    }

    // Empty constructor for flexibility
    public OptimalPowerGridSolution() {
    }

    // Getter for the maximum number of satisfied demands
    public int getMaxNumberOfSatisfiedDemands() {
        return maxNumberOfSatisfiedDemands;
    }

    // Getter for the hours to discharge batteries for maximum efficiency
    public ArrayList<Integer> getHoursToDischargeBatteriesForMaxEfficiency() {
        return hoursToDischargeBatteriesForMaxEfficiency;
    }

    // Getter for the total demanded gigawatts
    public int getTotalDemandedGigawatts() {
        return totalDemandedGigawatts;
    }

    // Getter for the number of unsatisfied gigawatts
    public int getUnsatisfiedGigawatts() {
        return unsatisfiedGigawatts;
    }

    // Additional setters if modification is required
    public void setMaxNumberOfSatisfiedDemands(int maxNumberOfSatisfiedDemands) {
        this.maxNumberOfSatisfiedDemands = maxNumberOfSatisfiedDemands;
    }

    public void setHoursToDischargeBatteriesForMaxEfficiency(ArrayList<Integer> hoursToDischargeBatteriesForMaxEfficiency) {
        this.hoursToDischargeBatteriesForMaxEfficiency = hoursToDischargeBatteriesForMaxEfficiency;
    }

    public void setTotalDemandedGigawatts(int totalDemandedGigawatts) {
        this.totalDemandedGigawatts = totalDemandedGigawatts;
    }

    public void setUnsatisfiedGigawatts(int unsatisfiedGigawatts) {
        this.unsatisfiedGigawatts = unsatisfiedGigawatts;
    }
}
