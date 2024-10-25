public class LinearSearch {

    public static int linearSearch(int[] A, int x) {
        int size = A.length;
        for (int i = 0; i < size; i++) {
            if (A[i] == x) {
                return i;
            }
        }
        return -1;
    }
}