import java.util.Arrays;

public class Utils {
    public static Double[] sortingRun(Integer[][] subsets) {

        Double[] selectionSortTimes = new Double[10];
        Double[] quickSortTimes = new Double[10];
        Double[] bucketSortTimes = new Double[10];

        // sort every subset 10 times using selectionsort and calculate the average
        // runtime and save it in an array
        for (int i = 0; i < subsets.length; i++) {
            Double runtime = 0.0;

            for (int k = 0; k < 10; k++) {
                long startTime = System.nanoTime();
                SortingAlgorithms.selectionSort(subsets[i], subsets[i].length);
                long endTime = System.nanoTime();
                runtime += (double) (endTime - startTime);
            }

            selectionSortTimes[i] = runtime / 10;
        }

        return selectionSortTimes;
    }
}
