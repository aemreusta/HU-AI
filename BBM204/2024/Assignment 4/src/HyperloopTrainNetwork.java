import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class HyperloopTrainNetwork implements Serializable {
    static final long serialVersionUID = 11L;
    public double averageTrainSpeed;
    public final double averageWalkingSpeed = 1000 / 6.0;;
    public int numTrainLines;
    public Station startPoint;
    public Station destinationPoint;
    public List<TrainLine> lines;

    /**
     * Method with a Regular Expression to extract integer numbers from the fileContent
     * @return the result as int
     */
    public int getIntVar(String varName, String fileContent) {
        Pattern p = Pattern.compile("[\\t ]*" + varName + "[\\t ]*=[\\t ]*([0-9]+)");
        Matcher m = p.matcher(fileContent);
        m.find();
        return Integer.parseInt(m.group(1));
    }

    /**
     * Write the necessary Regular Expression to extract string constants from the fileContent
     * @return the result as String
     */
    public String getStringVar(String varName, String fileContent) {
        Pattern p = Pattern.compile("\\b" + varName + "\\s*=\\s*\"([^\"]+)\"");
        Matcher m = p.matcher(fileContent);
        if (m.find()) {
            return m.group(1);
        }
        return ""; // Return empty string if no match found
    }
    
    

    /**
     * Write the necessary Regular Expression to extract floating point numbers from the fileContent
     * Your regular expression should support floating point numbers with an arbitrary number of
     * decimals or without any (e.g. 5, 5.2, 5.02, 5.0002, etc.).
     * @return the result as Double
     */
    public Double getDoubleVar(String varName, String fileContent) {
        Pattern p = Pattern.compile("\\b" + varName + "\\s*=\\s*([0-9]*\\.?[0-9]+)");
        Matcher m = p.matcher(fileContent);
        if (m.find()) {
            return Double.parseDouble(m.group(1));
        }
        return 0.0; // Return 0.0 if no match found
    }
    
    

    /**
     * Write the necessary Regular Expression to extract a Point object from the fileContent
     * points are given as an x and y coordinate pair surrounded by parentheses and separated by a comma
     * @return the result as a Point object
     */
    public Point getPointVar(String varName, String fileContent) {
        Pattern p = Pattern.compile("\\b" + varName + "\\s*=\\s*\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)");
        Matcher m = p.matcher(fileContent);
        if (m.find()) {
            int x = Integer.parseInt(m.group(1));
            int y = Integer.parseInt(m.group(2));
            return new Point(x, y);
        }
        return new Point(0, 0); // Return default Point if no match found
    }
    
    
    /**
     * Function to extract the train lines from the fileContent by reading train line names and their 
     * respective stations.
     * @return List of TrainLine instances
     */
    public List<TrainLine> getTrainLines(String fileContent) {
        List<TrainLine> trainLines = new ArrayList<>();
        Pattern p = Pattern.compile("train_line_name\\s*=\\s*\"([^\"]+)\"\\s*train_line_stations\\s*=\\s*\\(([^)]+)\\)");
        Matcher m = p.matcher(fileContent);
    
        while (m.find()) {
            String lineName = m.group(1);
            String stationsData = m.group(2);
            List<Station> stations = new ArrayList<>();
    
            Matcher stationMatcher = Pattern.compile("\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)").matcher(stationsData);
            while (stationMatcher.find()) {
                int x = Integer.parseInt(stationMatcher.group(1));
                int y = Integer.parseInt(stationMatcher.group(2));
                stations.add(new Station(new Point(x, y), lineName));
            }
    
            trainLines.add(new TrainLine(lineName, stations));
        }
        return trainLines;
    }
    
    

    /**
     * Function to populate the given instance variables of this class by calling the functions above.
     */
    public void readInput(String filename) {
        try {
            // Read the entire content of the file into a single String
            String fileContent = new String(Files.readAllBytes(Paths.get(filename)));
    
            // Extracting the number of train lines
            this.numTrainLines = getIntVar("num_train_lines", fileContent);
            // System.out.println("Number of Train Lines: " + numTrainLines); // Debug
    
            // Extracting the average train speed
            this.averageTrainSpeed = getDoubleVar("average_train_speed", fileContent);
            // System.out.println("Average Train Speed: " + averageTrainSpeed); // Debug
    
            // Extracting the starting point
            this.startPoint = new Station(getPointVar("starting_point", fileContent), "Starting Point");
            // System.out.println("Starting Point: " + startPoint); // Debug
    
            // Extracting the destination point
            this.destinationPoint = new Station(getPointVar("destination_point", fileContent), "Final Destination");
            // System.out.println("Destination Point: " + destinationPoint); // Debug
    
            // Extracting the train lines
            this.lines = getTrainLines(fileContent);
            // System.out.println("Train Lines: " + lines.size()); // Debug
    
        } catch (IOException e) {
            System.out.println("Failed to read the file: " + filename);
            e.printStackTrace();
        } catch (Exception e) {
            System.out.println("An error occurred while parsing the file: " + filename);
            e.printStackTrace();
        }
    }
}