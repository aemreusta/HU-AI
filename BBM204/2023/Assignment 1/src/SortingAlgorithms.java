import java.util.ArrayList;
import java.util.Collections;

public class SortingAlgorithms {

    public static void bucketSort(int[] arr) {
        int n = arr.length;
        int numberOfBuckets = (int) Math.sqrt(n);
        ArrayList<Integer>[] buckets = new ArrayList[numberOfBuckets];
        int max = getMax(arr);

        // initialize empty buckets
        for (int i = 0; i < numberOfBuckets; i++) {
            buckets[i] = new ArrayList<Integer>();
        }

        // insert elements into buckets based on hash function
        for (int i = 0; i < n; i++) {
            int bucketIndex = hash(arr[i], max, numberOfBuckets);
            buckets[bucketIndex].add(arr[i]);
        }

        // sort each bucket individually
        for (int i = 0; i < numberOfBuckets; i++) {
            Collections.sort(buckets[i]);
        }

        // concatenate all buckets into a single array
        int index = 0;
        for (int i = 0; i < numberOfBuckets; i++) {
            for (int j = 0; j < buckets[i].size(); j++) {
                arr[index++] = buckets[i].get(j);
            }
        }
    }

    private static int getMax(int[] arr) {
        int max = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
            }
        }
        return max;
    }

    private static int hash(int i, int max, int numberOfBuckets) {
        return (int) Math.floor((i / max) * (numberOfBuckets - 1));
    }

    public static void quickSort(int[] arr, int low, int high) {
        int[] stack = new int[high - low + 1];
        int top = -1;
        stack[++top] = low;
        stack[++top] = high;

        while (top >= 0) {
            high = stack[top--];
            low = stack[top--];
            int pivotIdx = partition(arr, low, high);
            if (pivotIdx - 1 > low) {
                stack[++top] = low;
                stack[++top] = pivotIdx - 1;
            }
            if (pivotIdx + 1 < high) {
                stack[++top] = pivotIdx + 1;
                stack[++top] = high;
            }
        }
    }

    public static int partition(int[] arr, int low, int high) {
        int pivot = arr[high];
        int i = low - 1;
        for (int j = low; j <= high - 1; j++) {
            if (arr[j] <= pivot) {
                i++;
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;
        return i + 1;
    }

    public static void selectionSort(Integer[] subsets, int n) {

        for (int i = 0; i < n - 1; i++) {
            int minIdx = i;
            for (int j = i + 1; j < n; j++) {
                if (subsets[j] < subsets[minIdx]) {
                    minIdx = j;
                }
            }
            if (minIdx != i) {
                int temp = subsets[minIdx];
                subsets[minIdx] = subsets[i];
                subsets[i] = temp;
            }
        }
    }

}
