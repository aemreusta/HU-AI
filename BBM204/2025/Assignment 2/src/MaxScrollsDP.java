import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class MaxScrollsDP {
    private final ArrayList<ArrayList<Integer>> safesDiscovered;

    public MaxScrollsDP(ArrayList<ArrayList<Integer>> safesDiscovered) {
        this.safesDiscovered = safesDiscovered;
    }

    public OptimalScrollSolution optimalSafeOpeningAlgorithm() {
        int T = safesDiscovered.size();
        Map<Integer, Integer> dp = new HashMap<>();
        dp.put(0, 0);

        for (ArrayList<Integer> safe : safesDiscovered) {
            Map<Integer, Integer> newDp = new HashMap<>();
            int Ci = safe.get(0);
            int Si = safe.get(1);

            for (Map.Entry<Integer, Integer> entry : dp.entrySet()) {
                int k = entry.getKey();
                int scrolls = entry.getValue();

                // Generate knowledge
                int newK = k + 5;
                if (newK <= 5 * T) {
                    newDp.put(newK, Math.max(newDp.getOrDefault(newK, 0), scrolls));
                }

                // Open safe if possible
                if (k >= Ci) {
                    int remainingK = k - Ci;
                    newDp.put(remainingK, Math.max(newDp.getOrDefault(remainingK, 0), scrolls + Si));
                }

                // Maintain state
                newDp.put(k, Math.max(newDp.getOrDefault(k, 0), scrolls));
            }
            dp = newDp;
        }

        int maxScrolls = dp.values().stream().max(Integer::compare).orElse(0);
        return new OptimalScrollSolution(safesDiscovered, maxScrolls);
    }
}