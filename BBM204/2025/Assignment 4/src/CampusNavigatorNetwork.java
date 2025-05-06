// CampusNavigatorNetwork.java
import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class CampusNavigatorNetwork implements Serializable {
    static final long serialVersionUID = 11L;
    public double averageCartSpeed; // Store as m/min
    // Average walking speed: 10 km/h = 10000 m / 60 min = 166.66... m/min
    public final double averageWalkingSpeed = 10000.0 / 60.0;
    public int numCartLines;
    public Station startPoint;
    public Station destinationPoint;
    public List<CartLine> lines;
    private String fileContent; // Store file content for regex matching

    // Map to ensure unique station objects per coordinate/description
    private transient Map<String, Station> stationRegistry = new HashMap<>();

    private Station getOrCreateStation(Point p, String description) {
        // Use description as the primary key for uniqueness
        return stationRegistry.computeIfAbsent(description, k -> new Station(p, description));
    }
     private Station getOrCreateStation(int x, int y, String description) {
        return getOrCreateStation(new Point(x,y), description);
    }


    /**
     * Write the necessary Regular Expression to extract string constants from the fileContent
     * @return the result as String, or null if not found
     */
    public String getStringVar(String varName, String content) {
        // Regex allows for optional whitespace and captures the quoted string
        Pattern p = Pattern.compile("^[\\t ]*" + Pattern.quote(varName) + "[\\t ]*=[\\t ]*\"([^\"]*)\"", Pattern.MULTILINE);
        Matcher m = p.matcher(content);
        if (m.find()) {
            return m.group(1);
        }
        System.err.println("Warning: Could not find String variable: " + varName);
        return null;
    }

    /**
     * Write the necessary Regular Expression to extract floating point numbers from the fileContent
     * Supports formats like 5, 5.2, 5.0002.
     * @return the result as Double, or null if not found
     */
    public Double getDoubleVar(String varName, String content) {
         // Regex allows optional whitespace, captures floating point number
        Pattern p = Pattern.compile("^[\\t ]*" + Pattern.quote(varName) + "[\\t ]*=[\\t ]*([0-9]+\\.?[0-9]*|[0-9]*\\.?[0-9]+)", Pattern.MULTILINE);
        Matcher m = p.matcher(content);
        if (m.find()) {
            try {
                return Double.parseDouble(m.group(1));
            } catch (NumberFormatException e) {
                 System.err.println("Error parsing Double for variable: " + varName + ", value: " + m.group(1));
                return null;
            }
        }
        System.err.println("Warning: Could not find Double variable: " + varName);
        return null;
    }

    /** Pre-provided getIntVar - slightly modified for consistency and robustness */
    public Integer getIntVar(String varName, String content) {
         // Regex allows optional whitespace, captures integer
        Pattern p = Pattern.compile("^[\\t ]*" + Pattern.quote(varName) + "[\\t ]*=[\\t ]*([0-9]+)", Pattern.MULTILINE);
        Matcher m = p.matcher(content);
         if (m.find()) {
             try {
                return Integer.parseInt(m.group(1));
             } catch (NumberFormatException e) {
                 System.err.println("Error parsing Integer for variable: " + varName + ", value: " + m.group(1));
                 return null;
             }
         }
         System.err.println("Warning: Could not find Integer variable: " + varName);
         return null;
    }

    /**
     * Write the necessary Regular Expression to extract a Point object from the fileContent
     * points are given as (x, y)
     * @return the result as a Point object, or null if not found
     */
    public Point getPointVar(String varName, String content) {
        // Regex allows optional whitespace around numbers and comma, captures x and y
        Pattern p = Pattern.compile("^[\\t ]*" + Pattern.quote(varName) + "[\\t ]*=[\\t ]*\\(\\s*(-?\\d+)\\s*,\\s*(-?\\d+)\\s*\\)", Pattern.MULTILINE);
        Matcher m = p.matcher(content);
        if (m.find()) {
            try {
                int x = Integer.parseInt(m.group(1));
                int y = Integer.parseInt(m.group(2));
                return new Point(x, y);
            } catch (NumberFormatException e) {
                 System.err.println("Error parsing Point for variable: " + varName + ", value: (" + m.group(1) + "," + m.group(2) + ")");
                return null;
            }
        }
         System.err.println("Warning: Could not find Point variable: " + varName);
        return null;
    }

    /**
     * Function to extract the cart lines from the fileContent by reading train line names and their
     * respective stations. Uses the stationRegistry to ensure unique Station objects.
     * @return List of CartLine instances
     */
    public List<CartLine> getCartLines(String content) {
        List<CartLine> cartLines = new ArrayList<>();
        // Regex to find a line name followed by its stations on potentially subsequent lines
        // DOTALL allows . to match line breaks. MULTILINE allows ^ and $ to match line starts/ends.
        Pattern linePattern = Pattern.compile(
            "^[\\t ]*cart_line_name\\s*=\\s*\"([^\"]*)\"\\s*^" + // Capture line name (Group 1)
            "[\\t ]*cart_line_stations\\s*=\\s*(.*?)$", // Capture stations string (Group 2), non-greedy
            Pattern.MULTILINE | Pattern.DOTALL);

        // Regex to find individual points within the stations string
        Pattern pointPattern = Pattern.compile("\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)");

        Matcher lineMatcher = linePattern.matcher(content);
        int lastMatchEnd = 0;

        while (lineMatcher.find(lastMatchEnd)) { // Start search from end of last match
            String lineName = lineMatcher.group(1).trim();
            String stationsStr = lineMatcher.group(2).trim();
            // System.out.println("Found Line: " + lineName + " Stations: " + stationsStr); // Debug

            List<Station> stations = new ArrayList<>();
            Matcher pointMatcher = pointPattern.matcher(stationsStr);
            int stationIndex = 1;
            while (pointMatcher.find()) {
                try {
                    int x = Integer.parseInt(pointMatcher.group(1));
                    int y = Integer.parseInt(pointMatcher.group(2));
                    String stationDesc = lineName + " Station " + stationIndex++;
                    // Use registry to get/create unique station object
                    stations.add(getOrCreateStation(x, y, stationDesc));
                } catch (NumberFormatException e) {
                    System.err.println("Error parsing station coordinates in line: " + lineName + ", value: " + pointMatcher.group(0));
                }
            }

            if (!stations.isEmpty()) {
                cartLines.add(new CartLine(lineName, stations));
            } else {
                System.err.println("Warning: No valid stations found for cart line: " + lineName);
            }
            lastMatchEnd = lineMatcher.end(); // Update search position
        }

         if (cartLines.isEmpty() && numCartLines > 0) {
            System.err.println("Warning: Expected " + numCartLines + " cart lines but found none. Check input format.");
        }

        return cartLines;
    }


    /**
     * Function to populate the given instance variables of this class by calling the functions above.
     * Returns true on success, false on failure.
     */
    public boolean readInput(String filename) {
        stationRegistry = new HashMap<>(); // Reset registry for each read
        try {
            this.fileContent = Files.readString(Paths.get(filename));

            Integer numLines = getIntVar("num_cart_lines", fileContent);
            Point startCoords = getPointVar("starting_point", fileContent);
            Point destCoords = getPointVar("destination_point", fileContent);
            Double cartSpeedKmh = getDoubleVar("average_cart_speed", fileContent);

            // Basic validation
            if (numLines == null || startCoords == null || destCoords == null || cartSpeedKmh == null) {
                System.err.println("Error: Missing one or more required variables in " + filename);
                return false;
            }

            this.numCartLines = numLines;
            // Use registry for start/destination points as well
            this.startPoint = getOrCreateStation(startCoords, "Starting Point");
            this.destinationPoint = getOrCreateStation(destCoords, "Final Destination");

            // Convert speed from km/h to m/min
            this.averageCartSpeed = cartSpeedKmh * 1000.0 / 60.0;

            this.lines = getCartLines(fileContent);

            // Add start/dest to registry if not already there (should be covered by getOrCreateStation)
            stationRegistry.putIfAbsent(this.startPoint.description, this.startPoint);
            stationRegistry.putIfAbsent(this.destinationPoint.description, this.destinationPoint);


            return true; // Indicate success

        } catch (IOException e) {
            System.err.println("Error reading input file: " + filename);
            e.printStackTrace();
            return false; // Indicate failure
        }
    }

     // Helper to get all unique stations
    public Collection<Station> getAllStations() {
        return stationRegistry.values();
    }
}