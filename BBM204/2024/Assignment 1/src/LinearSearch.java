public class LinearSearch {

    // Method to perform linear search in an array
    public int search(int[] array, int value) {
        // Iterate over each element in the array
        for (int i = 0; i < array.length; i++) {
            // Check if the current element matches the search value
            if (array[i] == value) {
                return i; // Value found, return its index
            }
        }
        return -1; // Value not found, return -1
    }
}
