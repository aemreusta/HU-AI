import java.util.ArrayList;
import java.util.Collections;

public class BucketSort {

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

}
