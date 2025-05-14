import java.util.List;
import java.util.Locale;

public class Main {
    public static void main(String[] args) {

        // Set the default locale to English
        Locale locale = new Locale("en", "EN");
        Locale.setDefault(locale);

        System.out.println("### CLUB FAIR SETUP START ###");
        ClubFairSetupPlanner planner = new ClubFairSetupPlanner();
        List<Project> projectList = planner.readXML(args[0]);
        // List<Project> projectList = planner.readXML("/Users/emre/GitHub/HU-AI/BBM204/2025/Assignment 4/src/io/projects.xml");
        planner.printSchedule(projectList);
        System.out.println("### CLUB FAIR SETUP END ###");

        System.out.println("### CAMPUS NAVIGATOR START ###");
        CampusNavigatorApp navigatorApp = new CampusNavigatorApp();
        CampusNavigatorNetwork network = navigatorApp.readCampusNavigatorNetwork(args[1]);
        // CampusNavigatorNetwork network = navigatorApp.readCampusNavigatorNetwork("/Users/emre/GitHub/HU-AI/BBM204/2025/Assignment 4/src/io/campus_navigator.dat");
        // CampusNavigatorNetwork network = navigatorApp.readCampusNavigatorNetwork("/Users/emre/GitHub/HU-AI/BBM204/2025/Assignment 4/src/io/Campus Navigator Input 2.dat");


        // // Print all cart lines
        // System.out.println("### CART LINES ###");
        // for (CartLine cartLine : network.lines) {
        //     System.out.println("Cart Line: " + cartLine.cartLineName);
        //     System.out.println("Stations:");
        //     for (Station station : cartLine.cartLineStations) {
        //         System.out.println("  " + station.description + " at coordinates (" + station.coordinates.x + ", " + station.coordinates.y + ")");
        //     }
        //     System.out.println();  // Blank line for separation between cart lines
        // }
        
        List<RouteDirection> directions = navigatorApp.getFastestRouteDirections(network);
        navigatorApp.printRouteDirections(directions);
        System.out.println("### CAMPUS NAVIGATOR END ###");
    }
}
