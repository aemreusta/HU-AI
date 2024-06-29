import java.util.Arrays;

public class MergeSort {

    // Main method to sort an array using merge sort algorithm
    public int[] sort(int[] A) {
        int n = A.length;
        if (n <= 1) {
            return A; // Arrays with one element are already sorted
        }

        // Split the array into two halves
        int[] left = Arrays.copyOfRange(A, 0, n / 2);
        int[] right = Arrays.copyOfRange(A, n / 2, n);

        // Recursively sort both halves
        left = sort(left);
        right = sort(right);

        // Merge the sorted halves and return the result
        return merge(left, right);
    }

    // Merge two sorted arrays into a single sorted array
    private int[] merge(int[] A, int[] B) {
        int[] C = new int[A.length + B.length]; // Resultant array
        int i = 0, j = 0, k = 0; // Index counters for A, B, and C respectively

        // Merge elements of A and B into C until one of them runs out
        while (i < A.length && j < B.length) {
            if (A[i] <= B[j]) {
                C[k++] = A[i++];
            } else {
                C[k++] = B[j++];
            }
        }

        // Copy remaining elements of A, if any
        while (i < A.length) {
            C[k++] = A[i++];
        }

        // Copy remaining elements of B, if any
        while (j < B.length) {
            C[k++] = B[j++];
        }

        return C;
    }
}

