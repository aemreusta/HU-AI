public class HashTable {
    private int[] table;
    private int size;

    public HashTable(int size) {
        this.size = size;
        this.table = new int[size];
    }

    public void insert(int value) {
        int index = hash(value);
        int i = 1;
        int newIndex = 0;
        if (table[index] == 0) {
            table[index] = value;
        } else {
            while (table[newIndex] != 0) {
                newIndex = (index + i * i) % size;
                i++;
            }
            table[newIndex] = value;
        }
    }

    private int hash(int value) {
        return value % size;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < size; i++) {
            sb.append(i).append(":").append(table[i]);
            if (i < size) {
                sb.append("|");
            }
        }
        return sb.toString();
    }
}