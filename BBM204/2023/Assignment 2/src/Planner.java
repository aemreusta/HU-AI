import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;

public class Planner {

    public final Task[] taskArray;
    public final Integer[] compatibility;
    public final Double[] maxWeight;
    public final ArrayList<Task> planDynamic;
    public final ArrayList<Task> planGreedy;

    public Planner(Task[] taskArray) {

        // Should be instantiated with an Task array
        // All the properties of this class should be initialized here

        this.taskArray = taskArray;
        this.compatibility = new Integer[taskArray.length];
        this.maxWeight = new Double[taskArray.length];
        this.planDynamic = new ArrayList<>();
        this.planGreedy = new ArrayList<>();
    }

    /**
     * @param index of the {@link Task}
     * @return Returns the index of the last compatible {@link Task},
     *         returns -1 if there are no compatible {@link Task}s.
     */
    public int binarySearch(int index) {
        int start = 0, end = index - 1, mid;
        while (start <= end) {
            mid = start + (end - start) / 2;
            if (taskArray[mid].getFinishTime().compareTo(taskArray[index].start) <= 0) {
                if (taskArray[mid + 1].getFinishTime().compareTo(taskArray[index].start) <= 0)
                    start = mid + 1;
                else
                    return mid;
            }

            else
                end = mid - 1;
        }
        return -1;
    }

    /**
     * {@link #compatibility} must be filled after calling this method
     */
    public void calculateCompatibility() {

        for (int i = 0; i < taskArray.length; i++) {
            compatibility[i] = binarySearch(i);
        }

        // print compatibility array
        for (int i = 0; i < taskArray.length; i++) {
            System.out.println(taskArray[i].name + " -> compatibility[" + i + "] = " +
                    compatibility[i]);
        }
    }

    /**
     * Uses {@link #taskArray} property
     * This function is for generating a plan using the dynamic programming
     * approach.
     * 
     * @return Returns a list of planned tasks.
     */
    public ArrayList<Task> planDynamic() {
        Arrays.sort(taskArray);
        calculateCompatibility();
        for (int i = 0; i < taskArray.length; i++) {
            solveDynamic(i);
        }
        return planDynamic;
    }

    /**
     * {@link #planDynamic} must be filled after calling this method
     */
    public void solveDynamic(int i) {
        if (i == 0 || compatibility[i] == -1) {
            planDynamic.add(taskArray[i]);
        } else {
            Double weightWithI = taskArray[i].importance + calculateMaxWeight(i - 1);
            Double weightWithoutI = calculateMaxWeight(i - 1);
            if (weightWithI > weightWithoutI) {
                planDynamic.add(taskArray[i]);
                solveDynamic(compatibility[i]);
            } else {
                solveDynamic(i - 1);
            }
        }
    }

    /**
     * {@link #maxWeight} must be filled after calling this method
     */
    /*
     * This function calculates maximum weights and prints out whether it has been
     * called before or not
     */
    public Double calculateMaxWeight(int i) {

        // System.out.println("Called calculateMaxWeight(" + i + ")");
        Double maxWeightWithI = taskArray[i].importance + calculateMaxWeight(compatibility[i]);
        Double maxWeightWithoutI = calculateMaxWeight(i - 1);
        maxWeight[i] = Math.max(maxWeightWithI, maxWeightWithoutI);
        return maxWeight[i];
    }

    /**
     * {@link #planGreedy} must be filled after calling this method
     * Uses {@link #taskArray} property
     *
     * @return Returns a list of scheduled assignments
     */

    /*
     * This function is for generating a plan using the greedy approach.
     */
    public ArrayList<Task> planGreedy() {
        Arrays.sort(taskArray);

        planGreedy.add(taskArray[0]);

        for (int i = 1; i < taskArray.length; i++) {
            if (taskArray[i].start.compareTo(planGreedy.get(planGreedy.size() - 1).getFinishTime()) >= 0) {
                planGreedy.add(taskArray[i]);
            }
        }
        return planGreedy;
    }
}
