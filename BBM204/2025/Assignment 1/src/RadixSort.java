public class RadixSort {

    public static int[] sort(int[] array) {
        int max = findMax(array);
        int d = (int) Math.log10(max) + 1;  // Find the number of digits of the largest number

        for (int pos = 1; pos <= d; pos++) {
            int[] count = new int[10]; // Assuming decimal digits (0-9)
            int[] output = new int[array.length];
            int size = array.length;

            // Count occurrences of each digit
            for (int i = 0; i < size; i++) {
                int digit = getDigit(array[i], pos);
                count[digit]++;
            }

            // Compute cumulative count
            for (int i = 1; i < 10; i++) {
                count[i] += count[i - 1];
            }

            // Place elements in sorted order
            for (int i = size - 1; i >= 0; i--) {
                int digit = getDigit(array[i], pos);
                count[digit]--;
                output[count[digit]] = array[i];
            }

            // Copy the sorted elements back into the original array
            System.arraycopy(output, 0, array, 0, size);
        }
        return array;  // Return the sorted array
    }

    private static int getDigit(int number, int pos) {
        return (number / (int) Math.pow(10, pos - 1)) % 10;
    }

    private static int findMax(int[] array) {
        int max = array[0];
        for (int num : array) {
            if (num > max) {
                max = num;
            }
        }
        return max;
    }
}
