import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        int algae = input.nextInt(); // total grams of algae
        int days = input.nextInt(); // number of days available for cleaning the lake
        int type = input.nextInt(); // number of available different shrimp species

        Shrimp[] shrimps = new Shrimp[type];
        for (int i = 0; i < type; i++) {
            String[] values = input.next().split(",");
            int cost = Integer.parseInt(values[0]);
            int eats = Integer.parseInt(values[1]);
            int amount = Integer.parseInt(values[2]);
            shrimps[i] = new Shrimp(i + 1, cost, eats, amount);
        }

        Arrays.sort(shrimps, Comparator.comparingDouble(Shrimp::getEfficiency).reversed()); // sort by efficiency

        int totalEats = 0;
        int[] buys = new int[type];

        for (int i = 0; i < type; i++) {

            for (int j = 0; j < shrimps[i].amount; j++) {
                if (totalEats >= algae / days) {
                    break; // enough shrimps to clean the lake
                }
                totalEats += shrimps[i].eats;
                buys[i]++;
            }

        }

        if (totalEats < algae / days) {
            System.out.print("Infeasible"); // not enough shrimps to clean the lake
        }

        else {
            int totalCost = 0;

            for (int i = 0; i < buys.length; i++) {
                totalCost += buys[i] * shrimps[i].cost;

                if (buys[i] > 0) {
                    System.out.println("Bought " + buys[i] + " of shrimp " + shrimps[i].type + ".");
                }

            }

            System.out.print("Total: " + totalCost + "$.");
        }
    }

    static class Shrimp {
        int type;
        int cost;
        int eats;
        int amount;

        Shrimp(int type, int cost, int eats, int amount) {
            this.type = type;
            this.cost = cost;
            this.eats = eats;
            this.amount = amount;
        }

        double getEfficiency() {
            return (double) eats / cost;
        }
    }
}
