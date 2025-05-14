import java.io.BufferedReader;
import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class CampusNavigatorNetwork implements Serializable {
    static final long serialVersionUID = 11L;
    public double averageCartSpeed;
    public final double averageWalkingSpeed = 1000.0 / 6.0; // 10 km/h = 10000m / 60min = 1000/6 m/min
    public int numCartLines;
    public Station startPoint;
    public Station destinationPoint;
    public List<CartLine> lines;

    public int getIntVar(String varName, String fileContent) {
        Pattern p = Pattern.compile("[\\t ]*" + varName + "[\\t ]*=[\\t ]*([0-9]+)");
        Matcher m = p.matcher(fileContent);
        if (m.find()) {
            return Integer.parseInt(m.group(1));
        } else {
            throw new IllegalArgumentException("Could not find integer variable: " + varName);
        }
    }

    public double getDoubleVar(String varName, String fileContent) {
        Pattern p = Pattern.compile("[\\t ]*" + varName + "[\\t ]*=[\\t ]*([0-9]*\\.?[0-9]+)");
        Matcher m = p.matcher(fileContent);
        if (m.find()) {
            return Double.parseDouble(m.group(1));
        } else {
            throw new IllegalArgumentException("Could not find double variable: " + varName);
        }
    }

    public String getStringVar(String varName, String fileContent) {
        Pattern p = Pattern.compile("\\b" + varName + "\\s*=\\s*\"([^\"]+)\"");
        Matcher m = p.matcher(fileContent);
        if (m.find()) {
            return m.group(1);
        } else {
            throw new IllegalArgumentException("Could not find string variable: " + varName);
        }
    }

    public Point getPointVar(String varName, String fileContent) {
        Pattern p = Pattern.compile("\\b" + varName + "\\s*=\\s*\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)");
        Matcher m = p.matcher(fileContent);
        if (m.find()) {
            int x = Integer.parseInt(m.group(1));
            int y = Integer.parseInt(m.group(2));
            return new Point(x, y);
        } else {
            throw new IllegalArgumentException("Could not find point variable: " + varName);
        }
    }

    public List<CartLine> getCartLines(String fileContent) {
        List<CartLine> cartLinesList = new ArrayList<>();
        Pattern p = Pattern.compile("cart_line_name\\s*=\\s*\"([^\"]+)\"\\s*cart_line_stations\\s*=\\s*((?:\\(\\s*\\d+\\s*,\\s*\\d+\\s*\\)\\s*)+)");
        Matcher m = p.matcher(fileContent);

        while (m.find()) {
            String lineName = m.group(1);
            String stationsData = m.group(2);
            List<Station> stations = new ArrayList<>();
            int stationIndex = 1;
            Matcher stationMatcher = Pattern.compile("\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)").matcher(stationsData);
            while (stationMatcher.find()) {
                int x = Integer.parseInt(stationMatcher.group(1));
                int y = Integer.parseInt(stationMatcher.group(2));
                String stationName = lineName + " Station " + stationIndex++;
                stations.add(new Station(new Point(x, y), stationName));
            }
            cartLinesList.add(new CartLine(lineName, stations));
        }
        return cartLinesList;
    }

    public void readInput(String filename) {
        try {
            String fileContent = new String(Files.readAllBytes(Paths.get(filename)));
            this.numCartLines = getIntVar("num_cart_lines", fileContent);
            
            this.averageCartSpeed = getIntVar("average_cart_speed", fileContent) * 1000 / 60.0; // Convert to m/min
            
            this.startPoint = new Station(getPointVar("starting_point", fileContent), "Starting Point");
            
            this.destinationPoint = new Station(getPointVar("destination_point", fileContent), "Final Destination");
            
            this.lines = getCartLines(fileContent);

        } catch (IOException e) {
            System.out.println("Failed to read the file: " + filename);
            e.printStackTrace();
        } catch (Exception e) {
            System.out.println("An error occurred while parsing the file: " + filename);
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        CampusNavigatorNetwork network = new CampusNavigatorNetwork();
        network.readInput("/Users/emre/GitHub/HU-AI/BBM204/2025/Assignment 4/src/io/Campus Navigator Input 2.dat");
        // Now you can work with the network object, e.g., print the details
        System.out.println("Number of Cart Lines: " + network.numCartLines);
        System.out.println("Average Cart Speed (m/min): " + network.averageCartSpeed);
        System.out.println("Start Point: (" + network.startPoint.coordinates.x + ", " + network.startPoint.coordinates.y + ")");
        System.out.println("Destination Point: (" + network.destinationPoint.coordinates.x + ", " + network.destinationPoint.coordinates.y + ")");
        for (CartLine line : network.lines) {
            System.out.println("Cart Line: " + line.cartLineName);
            for (Station station : line.cartLineStations) {
                System.out.println("Station: " + station.description + " at (" + station.coordinates.x + ", " + station.coordinates.y + ")");
            }
        }
    }
}