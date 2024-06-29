import java.util.List;

public class Main {
    public static void main(String[] args) {

        args = new String[2];

        args[0] = "/Users/emre/GitHub/HU-AI/BBM204/2024/Resit Assignment/input/groundwork_input_1.xml";

        args[1] = "/Users/emre/GitHub/HU-AI/BBM204/2024/Resit Assignment/input/exploration_input_1.xml";

        System.out.println("### MISSION GROUNDWORK START ###");
        MissionGroundwork missionGroundwork = new MissionGroundwork();
        List<Project> projectList = missionGroundwork.readXML(args[0]);
        missionGroundwork.printSchedule(projectList);
        System.out.println("### MISSION GROUNDWORK END ###");

        System.out.println("### MISSION EXPLORATION START ###");
        MissionExploration missionExploration = new MissionExploration();
        Galaxy galaxy = missionExploration.readXML(args[1]);
        missionExploration.printSolarSystems(galaxy);
        System.out.println("### MISSION EXPLORATION END ###");
    }
}

