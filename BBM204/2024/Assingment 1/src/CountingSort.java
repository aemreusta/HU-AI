import java.util.Arrays;

public class CountingSort {

    // Method to perform counting sort on an array A.
    public int[] sort(int[] A) {
        // Find the maximum and minimum values in A to determine the range.
        int max = Arrays.stream(A).max().getAsInt();
        int min = Arrays.stream(A).min().getAsInt();
        int range = max - min + 1;

        // Initialize the count array to store the count of each number
        // and the output array for the sorted numbers.
        int[] count = new int[range];
        int[] output = new int[A.length];

        // Count each number's occurrences in the input array.
        for (int num : A) {
            count[num - min]++;
        }

        // Accumulate the count array such that each element at each index 
        // stores the sum of previous counts. This modifies the count array
        // to contain the actual position of the elements in sorted order.
        for (int i = 1; i < range; i++) {
            count[i] += count[i - 1];
        }

        // Build the output array by placing the elements in their correct
        // positions and decreasing their count by one.
        for (int i = A.length - 1; i >= 0; i--) {
            output[count[A[i] - min] - 1] = A[i];
            count[A[i] - min]--;
        }

        return output; // Return the sorted array.
    }
}
