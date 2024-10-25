import java.util.*;
import java.io.*;

public class Quiz1 {
    public static void main(String[] args) throws IOException {
        // Setup to read from a file specified in the command line arguments
        BufferedReader reader = new BufferedReader(new FileReader(args[0]));
        // Use a HashSet to store words to ignore for efficient lookup
        Set<String> ignoreWords = new HashSet<>();
        // List to store the titles read from the input file
        List<String> titles = new ArrayList<>();
        String line;
        // Flag to distinguish between reading ignore words and titles
        boolean readingTitles = false;

        // Read each line of the input file
        while ((line = reader.readLine()) != null) {
            // Trim the line to remove any leading or trailing whitespace
            line = line.trim();
            // Check for the delimiter to switch from reading ignore words to titles
            if (line.equals("...")) {
                readingTitles = true;
                continue;
            }
            // Depending on the flag, add the line to either the ignore list or the titles list
            if (readingTitles) {
                titles.add(line);
            } else {
                // Convert ignore words to lowercase for case-insensitive comparison
                ignoreWords.add(line.toLowerCase());
            }
        }
        // Close the BufferedReader
        reader.close();

        // TreeMap to store the titles sorted by keywords, ensuring alphabetical order
        TreeMap<String, List<String>> sortedTitles = new TreeMap<>();
        for (String title : titles) {
            // Split the title into words by spaces, handling multiple consecutive spaces
            List<String> words = Arrays.asList(title.split("\\s+"));
            for (int i = 0; i < words.size(); i++) {
                // Normalize the current word for comparison against the ignore list
                String word = words.get(i).toLowerCase();
                // Check if the word is not in the ignore list to treat it as a keyword
                if (!ignoreWords.contains(word)) {
                    // Create a version of the title with the current keyword in uppercase
                    String modifiedTitle = createModifiedTitle(words, i);
                    // Add the modified title to the map, creating a new list if necessary
                    sortedTitles.computeIfAbsent(words.get(i).toUpperCase(), k -> new ArrayList<>()).add(modifiedTitle);
                }
            }
        }

        // Output the sorted titles by iterating over the TreeMap
        for (Map.Entry<String, List<String>> entry : sortedTitles.entrySet()) {
            for (String title : entry.getValue()) {
                System.out.println(title);
            }
        }
    }

    // Helper method to create a title string with the specified keyword in uppercase
    private static String createModifiedTitle(List<String> words, int index) {
        List<String> modifiedWords = new ArrayList<>();
        for (int i = 0; i < words.size(); i++) {
            // Convert the keyword to uppercase and keep other words in lowercase
            if (i == index) {
                modifiedWords.add(words.get(i).toUpperCase());
            } else {
                modifiedWords.add(words.get(i).toLowerCase());
            }
        }
        // Join the words back into a single title string
        return String.join(" ", modifiedWords);
    }
}
