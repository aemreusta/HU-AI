import java.io.*;
import java.util.*;

public class Quiz4 {

    /**
     * A class representing a node in the Trie data structure.
     */
    static class TrieNode {
        Map<Character, TrieNode> children = new HashMap<>();
        List<WordWeight> wordWeights = new ArrayList<>();
    }

    /**
     * A class representing a word and its associated weight.
     */
    static class WordWeight {
        String word;
        long weight;

        WordWeight(String word, long weight) {
            this.word = word;
            this.weight = weight;
        }
    }

    /**
     * A class representing a Trie data structure.
     */
    static class Trie {
        TrieNode root = new TrieNode();

        /**
         * Inserts a word with its associated weight into the Trie.
         *
         * @param word   the word to insert
         * @param weight the weight of the word
         */
        public void insert(String word, long weight) {
            TrieNode currentNode = root;
            for (char character : word.toCharArray()) {
                currentNode.children.putIfAbsent(character, new TrieNode());
                currentNode = currentNode.children.get(character);
            }
            currentNode.wordWeights.add(new WordWeight(word, weight));
        }

        /**
         * Collects all word-weight pairs from the given node and its descendants.
         *
         * @param node    the starting node
         * @param results the list to store collected word-weight pairs
         */
        private void collectAllWords(TrieNode node, List<WordWeight> results) {
            if (node != null) {
                results.addAll(node.wordWeights);
                for (TrieNode childNode : node.children.values()) {
                    collectAllWords(childNode, results);
                }
            }
        }

        /**
         * Searches for words with the given prefix and returns up to the specified limit
         * of results, sorted by weight in descending order.
         *
         * @param prefix the prefix to search for
         * @param limit  the maximum number of results to return
         * @return a list of word-weight pairs matching the prefix
         */
        public List<WordWeight> search(String prefix, int limit) {
            TrieNode currentNode = root;
            for (char character : prefix.toCharArray()) {
                if (!currentNode.children.containsKey(character)) {
                    return new ArrayList<>();
                }
                currentNode = currentNode.children.get(character);
            }
            List<WordWeight> results = new ArrayList<>();
            collectAllWords(currentNode, results);
            results.sort((wordWeight1, wordWeight2) -> Long.compare(wordWeight2.weight, wordWeight1.weight));
            if (limit > 0 && results.size() > limit) {
                return results.subList(0, limit);
            }
            return results;
        }
    }

    public static void main(String[] args) {
        try {
            Trie trie = new Trie();
            loadDatabase(trie, args[0]);
            processQueries(trie, args[1]);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Loads the database file into the Trie.
     *
     * @param trie             the Trie to load the data into
     * @param databaseFilePath the path to the database file
     * @throws IOException if an I/O error occurs
     */
    private static void loadDatabase(Trie trie, String databaseFilePath) throws IOException {
        try (BufferedReader databaseReader = new BufferedReader(new FileReader(databaseFilePath))) {
            int numberOfEntries = Integer.parseInt(databaseReader.readLine());
            for (int i = 0; i < numberOfEntries; i++) {
                String[] entry = databaseReader.readLine().split("\t");
                long wordWeight = Long.parseLong(entry[0]);
                String word = entry[1].toLowerCase();
                trie.insert(word, wordWeight);
            }
        }
    }

    /**
     * Processes the query file and prints the search results.
     *
     * @param trie          the Trie to search in
     * @param queryFilePath the path to the query file
     * @throws IOException if an I/O error occurs
     */
    private static void processQueries(Trie trie, String queryFilePath) throws IOException {
        try (BufferedReader queryReader = new BufferedReader(new FileReader(queryFilePath))) {
            String queryLine;
            while ((queryLine = queryReader.readLine()) != null) {
                String[] query = queryLine.split("\t");
                String prefix = query[0].toLowerCase();
                int resultLimit = Integer.parseInt(query[1]);
                List<WordWeight> results = trie.search(prefix, resultLimit);
                printResults(prefix, resultLimit, results);
            }
        }
    }

    /**
     * Prints the search results.
     *
     * @param prefix      the search prefix
     * @param resultLimit the result limit
     * @param results     the list of word-weight pairs
     */
    private static void printResults(String prefix, int resultLimit, List<WordWeight> results) {
        System.out.println("Query received: \"" + prefix + "\" with limit " + resultLimit + ". Showing results:");
        if (results.isEmpty() || resultLimit == 0) {
            System.out.println("No results.");
        } else {
            for (WordWeight wordWeight : results) {
                System.out.println("- " + wordWeight.weight + " " + wordWeight.word);
            }
        }
    }
}
