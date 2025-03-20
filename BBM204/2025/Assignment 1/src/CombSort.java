public class CombSort {
    
    public static void sort(int[] array) {
        int gap = array.length;
        double shrink = 1.3;
        boolean sorted = false;

        while (!sorted) {
            gap = Math.max(1, (int) Math.floor(gap / shrink));
            sorted = (gap == 1);

            for (int i = 0; i + gap < array.length; i++) {
                if (array[i] > array[i + gap]) {
                    // Swap elements
                    int temp = array[i];
                    array[i] = array[i + gap];
                    array[i + gap] = temp;

                    sorted = false;
                }
            }
        }
    }
}