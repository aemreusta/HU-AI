import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class MaxScrollsDP {
    private ArrayList<ArrayList<Integer>> safesDiscovered = new ArrayList<>();

    public MaxScrollsDP(ArrayList<ArrayList<Integer>> safesDiscovered) {
        this.safesDiscovered = safesDiscovered;
    }

    public ArrayList<ArrayList<Integer>> getSafesDiscovered() {
        return safesDiscovered;
    }

    public OptimalScrollSolution optimalSafeOpeningAlgorithm() {
        int T = safesDiscovered.size();
        if (T == 0) {
            return new OptimalScrollSolution(safesDiscovered, 0);
        }

        int maxKnowledge = 5 * T;
        Map<Integer, Integer> previous = new HashMap<>();
        previous.put(0, 0);

        for (ArrayList<Integer> safe : safesDiscovered) {
            int Ci = safe.get(0);
            int Si = safe.get(1);
            Map<Integer, Integer> current = new HashMap<>();

            for (Map.Entry<Integer, Integer> entry : previous.entrySet()) {
                int k_prev = entry.getKey();
                int scrolls_prev = entry.getValue();

                // Generate option
                int new_k_gen = k_prev + 5;
                if (new_k_gen <= maxKnowledge) {
                    if (current.containsKey(new_k_gen)) {
                        if (scrolls_prev > current.get(new_k_gen)) {
                            current.put(new_k_gen, scrolls_prev);
                        }
                    } else {
                        current.put(new_k_gen, scrolls_prev);
                    }
                }

                // Open option
                if (k_prev >= Ci) {
                    int new_k_open = k_prev - Ci;
                    int new_scrolls = scrolls_prev + Si;
                    if (current.containsKey(new_k_open)) {
                        if (new_scrolls > current.get(new_k_open)) {
                            current.put(new_k_open, new_scrolls);
                        }
                    } else {
                        current.put(new_k_open, new_scrolls);
                    }
                }

                // Maintain option
                if (current.containsKey(k_prev)) {
                    if (scrolls_prev > current.get(k_prev)) {
                        current.put(k_prev, scrolls_prev);
                    }
                } else {
                    current.put(k_prev, scrolls_prev);
                }
            }

            previous = new HashMap<>(current);
        }

        int maxScrolls = 0;
        for (int scrolls : previous.values()) {
            if (scrolls > maxScrolls) {
                maxScrolls = scrolls;
            }
        }

        return new OptimalScrollSolution(safesDiscovered, maxScrolls);
    }
}