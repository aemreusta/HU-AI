import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class CampusNavigatorNetwork implements Serializable {
    static final long serialVersionUID = 11L;
    public double averageCartSpeed; // Keep in km/h initially for parsing
    // Convert walking speed (10 km/h) to meters per minute
    public final double averageWalkingSpeed_mpm = 10.0 * 1000.0 / 60.0; // Approx 166.67 m/min
    public int numCartLines;
    public Station startPoint;
    public Station destinationPoint;
    public List<CartLine> lines;

    // Store file content for regex methods
    private transient String fileContent;

    /**
     * Write the necessary Regular Expression to extract string constants from the fileContent
     * @return the result as String, or null if not found
     */
    public String getStringVar(String varName, String content) {
        // Regex allows optional whitespace around '=', and captures content within quotes
        // Uses non-greedy quantifier *? inside quotes if needed, but [^"]* is usually sufficient
        Pattern p = Pattern.compile("[\\t ]*" + Pattern.quote(varName) + "[\\t ]*=[\\t ]*\"([^\"]*)\"");
        Matcher m = p.matcher(content);
        if (m.find()) {
            return m.group(1);
        }
        System.err.println("Warning: Could not find String variable '" + varName + "'");
        return null; // Indicate not found
    }

    /**
     * Write the necessary Regular Expression to extract floating point numbers from the fileContent
     * Your regular expression should support floating point numbers with an arbitrary number of
     * decimals or without any (e.g. 5, 5.2, 5.02, 5.0002, etc.).
     * @return the result as Double, or null if not found
     */
    public Double getDoubleVar(String varName, String content) {
        // Regex allows optional whitespace, matches digits, optional decimal part
        Pattern p = Pattern.compile("[\\t ]*" + Pattern.quote(varName) + "[\\t ]*=[\\t ]*([0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)");
        Matcher m = p.matcher(content);
        if (m.find()) {
            try {
                return Double.parseDouble(m.group(1));
            } catch (NumberFormatException e) {
                 System.err.println("Warning: Could not parse Double value for '" + varName + "': " + m.group(1));
                return null;
            }
        }
         System.err.println("Warning: Could not find Double variable '" + varName + "'");
        return null; // Indicate not found
    }

    /** Provided **/
    public int getIntVar(String varName, String content) {
        Pattern p = Pattern.compile("[\\t ]*" + Pattern.quote(varName) + "[\\t ]*=[\\t ]*([0-9]+)");
        Matcher m = p.matcher(content);
        if (m.find()) {
             try {
                return Integer.parseInt(m.group(1));
            } catch (NumberFormatException e) {
                 System.err.println("Warning: Could not parse Integer value for '" + varName + "': " + m.group(1));
                return -1; // Or throw exception
            }
        }
        System.err.println("Warning: Could not find Integer variable '" + varName + "'");
        return -1; // Indicate error or not found
    }

    /**
     * Write the necessary Regular Expression to extract a Point object from the fileContent
     * points are given as an x and y coordinate pair surrounded by parentheses and separated by a comma
     * @return the result as a Point object, or null if not found
     */
    public Point getPointVar(String varName, String content) {
        // Regex allows optional whitespace around '()', ',', and numbers
        Pattern p = Pattern.compile("[\\t ]*" + Pattern.quote(varName) + "[\\t ]*=[\\t ]*\\([\\t ]*([0-9]+)[\\t ]*,[\\t ]*([0-9]+)[\\t ]*\\)");
        Matcher m = p.matcher(content);
        if (m.find()) {
            try {
                int x = Integer.parseInt(m.group(1));
                int y = Integer.parseInt(m.group(2));
                return new Point(x, y);
            } catch (NumberFormatException e) {
                 System.err.println("Warning: Could not parse Point coordinates for '" + varName + "': " + m.group(1) + "," + m.group(2));
                 return null;
            }
        }
         System.err.println("Warning: Could not find Point variable '" + varName + "'");
        return null; // Indicate not found
    }

    /**
     * Function to extract the cart lines from the fileContent by reading train line names and their
     * respective stations. Uses regex to find name and station pairs.
     * @return List of CartLine instances
     */
   public List<CartLine> getCartLines(String content) {
        List<CartLine> cartLines = new ArrayList<>();
        // Regex to find a line name and its corresponding station list string
        // Assumes name is immediately followed by stations definition (possibly after other vars)
        // Using Pattern.DOTALL to allow '.' to match newlines between name and stations
        // Using reluctant quantifier .*? to avoid overmatching
        Pattern linePattern = Pattern.compile(
            "cart_line_name[\\t ]*=[\\t ]*\"([^\"]*)\".*?" + // Capture name (Group 1)
            "cart_line_stations[\\t ]*=[\\t ]*((?:\\([\\t ]*\\d+[\\t ]*,[\\t ]*\\d+[\\t ]*\\)[\\t ]*)+)", // Capture station list (Group 2)
            Pattern.DOTALL // Allow '.' to match newline characters
        );

        // Regex to find individual points within the station list string
        Pattern pointPattern = Pattern.compile("\\([\\t ]*(\\d+)[\\t ]*,[\\t ]*(\\d+)[\\t ]*\\)");

        Matcher lineMatcher = linePattern.matcher(content);
        int stationIndex = 1; // For unique station names if needed

        while (lineMatcher.find()) {
            String lineName = lineMatcher.group(1);
            String stationsString = lineMatcher.group(2);
            List<Station> stations = new ArrayList<>();

            Matcher pointMatcher = pointPattern.matcher(stationsString);
            int pointIndex = 1;
            while (pointMatcher.find()) {
                try {
                    int x = Integer.parseInt(pointMatcher.group(1));
                    int y = Integer.parseInt(pointMatcher.group(2));
                    // Create a unique station name
                    String stationName = lineName + " Station " + pointIndex;
                    stations.add(new Station(new Point(x, y), stationName));
                    pointIndex++;
                } catch (NumberFormatException e) {
                    System.err.println("Warning: Could not parse point coordinates in line '" + lineName + "': " + pointMatcher.group(0));
                }
            }

            if (!stations.isEmpty()) {
                cartLines.add(new CartLine(lineName, stations));
            } else {
                 System.err.println("Warning: No valid stations found for cart line '" + lineName + "'");
            }
        }

         if (cartLines.isEmpty() && numCartLines > 0) {
              System.err.println("Warning: Expected " + numCartLines + " cart lines but found none. Check regex or input format.");
         }

        return cartLines;
    }


    /**
     * Function to populate the given instance variables of this class by calling the functions above.
     */
    public void readInput(String filename) {
        try {
            // Read entire file content
            this.fileContent = new String(Files.readAllBytes(Paths.get(filename)));

            // Parse variables using the content string
            this.numCartLines = getIntVar("num_cart_lines", fileContent);
            if (this.numCartLines < 0) throw new IOException("Could not read num_cart_lines");

            Point startCoords = getPointVar("starting_point", fileContent);
            if (startCoords == null) throw new IOException("Could not read starting_point");
            this.startPoint = new Station(startCoords, "Starting Point"); // Assign standard name

            Point destCoords = getPointVar("destination_point", fileContent);
             if (destCoords == null) throw new IOException("Could not read destination_point");
            this.destinationPoint = new Station(destCoords, "Final Destination"); // Assign standard name

            Double speed = getDoubleVar("average_cart_speed", fileContent);
             if (speed == null) throw new IOException("Could not read average_cart_speed");
            this.averageCartSpeed = speed; // Keep in km/h

            // Get cart lines using the content string
            this.lines = getCartLines(fileContent);

             // Validate number of lines found matches expected number
             if (this.lines.size() != this.numCartLines) {
                  System.err.println("Warning: Found " + this.lines.size() + " cart lines, but expected " + this.numCartLines);
             }


        } catch (IOException e) {
            System.err.println("Error reading or parsing campus network file '" + filename + "': " + e.getMessage());
            // Optionally, re-throw or exit, or ensure default values are safe
            this.numCartLines = 0;
            this.lines = new ArrayList<>();
            this.averageCartSpeed = 0;
            // startPoint and destinationPoint might be null, handle this in navigator
        }
    }

     // Helper method to get cart speed in meters per minute
     public double getCartSpeedMetersPerMinute() {
         return this.averageCartSpeed * 1000.0 / 60.0;
     }
}