import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int f = sc.nextInt(); // number of floors
        int c = sc.nextInt(); // number of columns
        int s = sc.nextInt(); // starting column on top floor
        sc.nextLine(); // consume newline character

        char[][] instructions = new char[f][c];
        for (int i = 0; i < f; i++) {
            String line = sc.nextLine();
            for (int j = 0; j < c; j++) {
                instructions[i][j] = line.charAt(j);
            }
        }

        int[] position = { 0, s - 1 }; // starting position
        int steps = 0;
        Set<String> visited = new HashSet<>();
        visited.add(Arrays.toString(position));
        Map<String, Integer> stepsMap = new HashMap<>();
        stepsMap.put(Arrays.toString(position), steps);
        while (true) {
            int row = position[0];
            int col = position[1];
            char instruction = instructions[row][col];
            switch (instruction) {
                case 'N':
                    row--;
                    break;
                case 'S':
                    row++;
                    break;
                case 'E':
                    col++;
                    break;
                case 'W':
                    col--;
                    break;
            }
            
            // check if new position is out of bounds
            if (row < 0 || row >= f || col < 0 || col >= c) {
                System.out.println((steps+1) + " step(s) to freedom. Yay!");
                return;
            }

            String posString = Arrays.toString(new int[] { row, col });
            if (visited.contains(posString)) {
                // we have entered a loop
                int loopStart = stepsMap.get(posString);
                int loopSize = steps - loopStart + 1;
                System.out.println(loopStart + " step(s) before getting stuck in a loop of " + loopSize + " step(s).");
                return;
            }

            visited.add(posString);
            stepsMap.put(posString, steps+1);
            position[0] = row;
            position[1] = col;
            steps++;
        }
    }
}
