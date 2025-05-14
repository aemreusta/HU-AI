import java.io.Serializable;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.nio.file.Files; // Needed for reading whole file
import java.nio.file.Paths;   // Needed for reading whole file
import java.io.IOException;

public class CampusNavigatorNetwork implements Serializable {
    static final long serialVersionUID = 11L;
    public double averageCartSpeed;
    public final double averageWalkingSpeed = 10000.0 / 60.0; // 10 km/h in m/min
    public int numCartLines;
    public Station startPoint;
    public Station destinationPoint;
    public List<CartLine> lines;

    // Custom exception for parsing failures
    static class ParseException extends RuntimeException {
        public ParseException(String message) {
            super(message);
        }
        public ParseException(String message, Throwable cause) {
            super(message, cause);
        }
    }
    
    private String extractGroup(Matcher m, String varName, String groupType, boolean optional) {
        if (m.find()) {
            try {
                return m.group(1);
            } catch (IllegalStateException | IndexOutOfBoundsException e) {
                throw new ParseException("Regex for " + varName + " ("+groupType+") matched, but capture group 1 is missing. Faulty regex pattern.", e);
            }
        }
        if (optional) return null; // Return null if optional and not found
        throw new ParseException("Required variable '" + varName + "' ("+groupType+") not found in input.");
    }


    // These methods operate on the entire fileContent string.
    public String getStringVar(String varName, String fileContent) {
        // Pattern allows for spaces around varName, '=', and quotes. varName is quoted for literal matching.
        // Allows empty string "" as a value
        Pattern p = Pattern.compile("[\\s]*" + Pattern.quote(varName) + "[\\s]*=[\\s]*\"([^\"]*)\"[\\s]*", Pattern.MULTILINE);
        Matcher m = p.matcher(fileContent);
        String result = extractGroup(m, varName, "String", false);
        return result.trim(); // Trim the captured group
    }

    public Double getDoubleVar(String varName, String fileContent) {
        Pattern p = Pattern.compile("[\\s]*" + Pattern.quote(varName) + "[\\s]*=[\\s]*([0-9]+\\.?[0-9]*|[0-9]*\\.?[0-9]+)[\\s]*", Pattern.MULTILINE);
        Matcher m = p.matcher(fileContent);
        String valueStr = extractGroup(m, varName, "Double", false);
        try {
            return Double.parseDouble(valueStr);
        } catch (NumberFormatException e) {
            throw new ParseException("Malformed double value for '" + varName + "': " + valueStr, e);
        }
    }

    public int getIntVar(String varName, String fileContent) {
        // This method was provided in the starter skeleton for getIntVar
        // Pattern p = Pattern.compile("[\\t ]*" + varName + "[\\t ]*=[\\t ]*([0-9]+)");
        // Adopting similar robust pattern:
        Pattern p = Pattern.compile("[\\s]*" + Pattern.quote(varName) + "[\\s]*=[\\s]*([0-9]+)[\\s]*", Pattern.MULTILINE);
        Matcher m = p.matcher(fileContent);
        String valueStr = extractGroup(m, varName, "Integer", false);
        try {
            return Integer.parseInt(valueStr);
        } catch (NumberFormatException e) {
            throw new ParseException("Malformed integer value for '" + varName + "': " + valueStr, e);
        }
    }
    
    public Point getPointVar(String varName, String fileContent) {
        Pattern p = Pattern.compile("[\\s]*" + Pattern.quote(varName) + "[\\s]*=[\\s]*\\(\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*\\)[\\s]*", Pattern.MULTILINE);
        Matcher m = p.matcher(fileContent);
        if (m.find()) {
            try {
                String xStr = m.group(1);
                String yStr = m.group(2);
                return new Point(Integer.parseInt(xStr), Integer.parseInt(yStr));
            } catch (NumberFormatException e) {
                throw new ParseException("Malformed coordinates for '" + varName + "': (" + m.group(1) + "," + m.group(2) + ")", e);
            } catch (IllegalStateException | IndexOutOfBoundsException e) { // Regex matched but groups missing
                 throw new ParseException("Regex for " + varName + " (Point) matched, but capture groups are missing. Faulty regex pattern.", e);
            }
        }
        throw new ParseException("Required variable '" + varName + "' (Point) not found in input.");
    }

    // This is the crucial method for parsing multiple cart lines from the WHOLE fileContent
    public List<CartLine> getCartLines(String fileContent) {
        List<CartLine> cartLines = new ArrayList<>();
        
        // Pattern to find a cart_line_name
        Pattern namePattern = Pattern.compile("cart_line_name\\s*=\\s*\"([^\"]+)\""); // Name cannot be empty, ensure it's not just ""
        Matcher nameMatcher = namePattern.matcher(fileContent);

        // Pattern to find the stations block associated with a name
        // It looks for 'cart_line_stations = ' followed by zero or more (x,y) pairs
        Pattern stationsPattern = Pattern.compile("cart_line_stations\\s*=\\s*((?:\\(\\s*\\d+\\s*,\\s*\\d+\\s*\\)\\s*)*)"); 
        
        // Pattern to find individual (x,y) pairs within a stations block
        Pattern singleStationPattern = Pattern.compile("\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)");

        int searchStartOffset = 0; // Where to start searching for the next cart_line_name

        while (nameMatcher.find(searchStartOffset)) {
            String lineName = nameMatcher.group(1).trim(); // Get and trim the cart line name
            
            if (lineName.isEmpty()) { // Skip if line name is empty after trim
                // System.err.println("Warning: Found empty cart_line_name at offset " + nameMatcher.start() + ". Skipping.");
                searchStartOffset = nameMatcher.end(); // Advance search past this empty name
                continue;
            }

            List<Station> stationsList = new ArrayList<>();
            
            // Now, search for the stationsPattern *specifically after* the current name's match
            Matcher stationsMatcherForThisLine = stationsPattern.matcher(fileContent);
            // We need to ensure that the stations we find are indeed the ones immediately following this name
            // and not stations for a much later line, if the file had interleaved irrelevant data (unlikely per problem).
            
            if (stationsMatcherForThisLine.find(nameMatcher.end())) {
                // Check if the found stations block starts reasonably close to the name.
                // A simple heuristic: if another cart_line_name appears before these stations, then these stations don't belong.
                int nextNameSearchRegionStart = nameMatcher.end();
                Matcher nextNameFinder = namePattern.matcher(fileContent);
                boolean anotherNameBeforeStations = false;
                if (nextNameFinder.find(nextNameSearchRegionStart)) { // Look for the *next* name
                    if (nextNameFinder.start() < stationsMatcherForThisLine.start()) {
                        anotherNameBeforeStations = true;
                    }
                }

                if (!anotherNameBeforeStations) {
                    String allStationsBlock = stationsMatcherForThisLine.group(1); // group(1) is the station coordinates block string
                    
                    if (allStationsBlock != null && !allStationsBlock.trim().isEmpty()) {
                        Matcher singleStationMatcher = singleStationPattern.matcher(allStationsBlock);
                        int stationIndex = 1;
                        while (singleStationMatcher.find()) {
                            try {
                                int x = Integer.parseInt(singleStationMatcher.group(1));
                                int y = Integer.parseInt(singleStationMatcher.group(2));
                                String stationDescription = lineName + " Station " + stationIndex++;
                                stationsList.add(new Station(new Point(x, y), stationDescription));
                            } catch (NumberFormatException e) {
                                // System.err.println("Warning: Malformed coordinate in stations for '" + lineName + "': " + singleStationMatcher.group(0) + ". Skipping station.");
                            }
                        }
                    }
                    // Enforce PDF rule: "at least two stations per line"
                    if (stationsList.size() >= 2) {
                        cartLines.add(new CartLine(lineName, stationsList));
                    } else {
                        // System.err.println("Warning: Cart line '" + lineName + "' parsed with " + stationsList.size() + " valid stations (requires >= 2). Not adding line.");
                    }
                    // The next search for a cart_line_name should start after the stations block we just processed.
                    searchStartOffset = stationsMatcherForThisLine.end();
                } else {
                    // Another name was found before these stations, so these stations don't belong to current lineName.
                    // The current lineName is orphaned.
                    // System.err.println("Warning: Cart line '" + lineName + "' found without subsequent stations before next line name. Skipping line.");
                    searchStartOffset = nameMatcher.end(); // Next name search starts after current name.
                }
            } else {
                // No 'cart_line_stations' block found anywhere after this 'cart_line_name'. Orphaned name.
                // System.err.println("Warning: Cart line '" + lineName + "' found without any subsequent station block in file. Skipping line.");
                searchStartOffset = nameMatcher.end(); // Advance search past this name
            }
        }
        return cartLines;
    }

    public void readInput(String filename) {
        this.lines = new ArrayList<>(); // Initialize in case of errors

        try {
            // Read the entire file content into a single string
            String fileContent = new String(Files.readAllBytes(Paths.get(filename)));
            
            // 1. Parse and assign the starting point, destination point, and average cart speed.
            // These use the helper methods that operate on the whole fileContent.
            this.numCartLines = getIntVar("num_cart_lines", fileContent);
            this.startPoint = new Station(getPointVar("starting_point", fileContent), "Starting Point");
            this.destinationPoint = new Station(getPointVar("destination_point", fileContent), "Final Destination");
            this.averageCartSpeed = getDoubleVar("average_cart_speed", fileContent);
            
            // 2. Read all cart lines from the input (not just the first one)
            // 3. Correctly parse all stations under each line, preserving their order
            // 4. Assign the correct names (e.g., "Cart Line A Station 1") to each station
            this.lines = getCartLines(fileContent); // This method handles points 2, 3, 4 for cart lines

            // Optional: Validate if the number of lines parsed matches numCartLines declared in the file.
            // The TA description for TestCampusNavigatorAppReadInput: "Number of cart lines (numCartLines)"
            // This implies the field `this.numCartLines` should be set directly from the file.
            // The `this.lines.size()` should be the count of *validly parsed* lines.
            // If the test expects them to be equal, and your getCartLines filters out lines that numCartLines counts,
            // that could be a mismatch.
            // For now, we assume numCartLines is just a declaration and lines.size() is what's actually usable.
            // If a strict match is needed:
            // if (this.numCartLines != this.lines.size()) {
            //     throw new ParseException("Declared num_cart_lines (" + this.numCartLines + 
            //                            ") does not match actual number of valid cart lines parsed (" + this.lines.size() + ").");
            // }

        } catch (IOException e) {
            // System.err.println("FATAL: IOException while reading input file: " + filename + " - " + e.getMessage());
            throw new RuntimeException("Failed to read input file: " + filename, e);
        } catch (ParseException e) { // Catch our custom parsing exceptions
            // System.err.println("FATAL: Parsing error in input file: " + filename + " - " + e.getMessage());
            throw new RuntimeException("Failed to parse input file: " + filename + ". Error: " + e.getMessage(), e);
        }
    }
}