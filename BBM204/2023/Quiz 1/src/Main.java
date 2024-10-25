import java.util.Scanner;

public class Main {

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int size = sc.nextInt();
        HashTable ht = new HashTable(size);
        while (sc.hasNextInt()) {
            int value = sc.nextInt();
            ht.insert(value);
        }
        System.out.println(ht.toString());
    }
}
