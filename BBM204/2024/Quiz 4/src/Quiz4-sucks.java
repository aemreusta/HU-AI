import java.io.*;
import java.util.*;

class TrieNode {
    Map<Character, TrieNode> children;
    boolean isEndOfWord;
    List<Result> results;  // List to store results with weights

    TrieNode() {
        children = new HashMap<>();
        isEndOfWord = false;
        results = new ArrayList<>();
    }
}

class Result {
    String word;
    int weight;

    Result(String word, int weight) {
        this.word = word;
        this.weight = weight;
    }
}

public class Quiz4 {

    private TrieNode root;

    public Quiz4() {
        root = new TrieNode();
    }

    public void insert(String word, int weight) {
        TrieNode current = root;
        for (char ch : word.toCharArray()) {
            current.children.putIfAbsent(ch, new TrieNode());
            current = current.children.get(ch);
        }
        current.isEndOfWord = true;
        current.results.add(new Result(word, weight));
    }

    public List<Result> search(String prefix) {
        TrieNode current = root;
        for (char ch : prefix.toCharArray()) {
            current = current.children.get(ch);
            if (current == null) {
                return new ArrayList<>();
            }
        }
        return current.results;
    }

    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            System.err.println("Usage: java Quiz4 <database_file> <query_file>");
            return;
        }

        Quiz4 trie = new Quiz4();

        // Read database file
        try (BufferedReader br = new BufferedReader(new FileReader(args[0]))) {
            int n = Integer.parseInt(br.readLine().trim());
            for (int i = 0; i < n; i++) {
                String[] parts = br.readLine().trim().split("\t");
                int weight = Integer.parseInt(parts[0]);
                String word = parts[1].toLowerCase();
                trie.insert(word, weight);
            }
        }

        // Read query file and process queries
        try (BufferedReader br = new BufferedReader(new FileReader(args[1]))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.trim().split("\t");
                String query = parts[0].toLowerCase();
                int limit = Integer.parseInt(parts[1]);

                System.out.println("Query received: \"" + query + "\" with limit " + limit + ". Showing results:");
                List<Result> results = trie.search(query);

                // Sort results by weight in descending order
                results.sort((a, b) -> Integer.compare(b.weight, a.weight));

                if (limit == 0 || results.isEmpty()) {
                    System.out.println("No results.");
                } else {
                    for (int i = 0; i < Math.min(limit, results.size()); i++) {
                        System.out.println("- " + results.get(i).weight + " " + results.get(i).word);
                    }
                }
            }
        }
    }
}
