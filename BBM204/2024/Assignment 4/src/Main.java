import java.util.List;
import java.util.Locale;

public class Main {
    public static void main(String[] args) {
        // create args array with the file paths
        // args = new String[2];
        // args[0] = "/Users/emre/GitHub/HU-AI/BBM204/2024/Assignment 4/inputs/part_1_sample_input.xml";

        // args[1] = "/Users/emre/GitHub/HU-AI/BBM204/2024/Assignment 4/inputs/part_2_sample_input.dat";

        // args[1] = "/Users/emre/GitHub/HU-AI/BBM204/2024/Assignment 4/inputs/urban_transportation_input_5.dat";

        Locale locale = new Locale("en_EN"); 
        Locale.setDefault(locale);

        System.out.println("### URBAN INFRASTRUCTURE DEVELOPMENT START ###");
        UrbanInfrastructureDevelopment urbanInfrastructureDevelopment = new UrbanInfrastructureDevelopment();
        List<Project> projectList = urbanInfrastructureDevelopment.readXML(args[0]);
        urbanInfrastructureDevelopment.printSchedule(projectList);
        System.out.println("### URBAN INFRASTRUCTURE DEVELOPMENT END ###");

        System.out.println("### URBAN TRANSPORTATION APP START ###");
        UrbanTransportationApp urbanTransportationApp = new UrbanTransportationApp();
        HyperloopTrainNetwork network = urbanTransportationApp.readHyperloopTrainNetwork(args[1]);
        List<RouteDirection> directions = urbanTransportationApp.getFastestRouteDirections(network);
        urbanTransportationApp.printRouteDirections(directions);
        System.out.println("### URBAN TRANSPORTATION APP END ###");

    }
}

