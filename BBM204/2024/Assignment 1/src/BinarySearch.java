class BinarySearch {

    // Method to perform binary search on a sorted array
    public int search(int[] array, int value) {
        int low = 0; // Starting index of the search range
        int high = array.length - 1; // Ending index of the search range

        // Continue searching while the search range is valid
        while (low <= high) {
            // Calculate the middle index of the current search range
            int mid = low + (high - low) / 2;

            // Check if the middle element is the target value
            if (array[mid] == value) {
                return mid; // Target value found, return its index
            } else if (array[mid] < value) {
                // Target value is in the upper half of the current search range
                low = mid + 1;
            } else {
                // Target value is in the lower half of the current search range
                high = mid - 1;
            }
        }
        // Target value not found in the array
        return -1;
    }
}
