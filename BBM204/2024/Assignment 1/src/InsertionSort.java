public class InsertionSort {

    // Optimized insertion sort method.
    public int[] sort(int[] A) {
        int n = A.length;
        // Directly modify the input array to avoid unnecessary cloning,
        // which reduces memory usage.
        for (int i = 1; i < n; i++) {
            int key = A[i];
            int j = i - 1;
            
            // Move elements of A[0..i-1], that are greater than key,
            // to one position ahead of their current position.
            while (j >= 0 && A[j] > key) {
                A[j + 1] = A[j];
                j = j - 1;
            }
            A[j + 1] = key;
        }
        return A; // Return the sorted array.
    }
}
