import java.io.File;
import java.util.*;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

public class MissionGroundwork {

    /**
     * Given a list of Project objects, prints the earliest schedule, latest schedule and mobility analysis of each of them.
     * Uses getEarliestSchedule(), getLatestSchedule(), getMobility(), printSchedule() and printMobility()  methods of the current project.
     * @param projectList a list of Project objects
     */
    public void printSchedule(List<Project> projectList) {
        for (Project project : projectList) {
            int[] earliestSchedule = project.getEarliestSchedule();
            project.printSchedule(earliestSchedule, "Earliest");
    
            int projectDuration = project.getProjectDuration();
            int[] latestSchedule = project.getLatestSchedule(projectDuration);
            project.printSchedule(latestSchedule, "Latest");
    
            int[] mobility = project.getMobility(earliestSchedule, latestSchedule);
            project.printMobility(mobility);
        }
    }
    

    /**
     * TODO: Parse the input XML file and return a list of Project objects
     *
     * @param filename the input XML file
     * @return a list of Project objects
     */

    public List<Project> readXML(String filename) {
        List<Project> projectList = new ArrayList<>();

        try {
            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            DocumentBuilder builder = factory.newDocumentBuilder();
            Document document = builder.parse(new File(filename));
            document.getDocumentElement().normalize();

            NodeList projectNodes = document.getElementsByTagName("Project");

            for (int i = 0; i < projectNodes.getLength(); i++) {
                Node projectNode = projectNodes.item(i);
                if (projectNode.getNodeType() == Node.ELEMENT_NODE) {
                    Element projectElement = (Element) projectNode;
                    String projectName = projectElement.getElementsByTagName("Name").item(0).getTextContent();
                    List<Task> taskList = new ArrayList<>();

                    NodeList taskNodes = projectElement.getElementsByTagName("Task");
                    for (int j = 0; j < taskNodes.getLength(); j++) {
                        Node taskNode = taskNodes.item(j);
                        if (taskNode.getNodeType() == Node.ELEMENT_NODE) {
                            Element taskElement = (Element) taskNode;
                            int taskID = Integer.parseInt(taskElement.getElementsByTagName("TaskID").item(0).getTextContent());
                            String description = taskElement.getElementsByTagName("Description").item(0).getTextContent();
                            int duration = Integer.parseInt(taskElement.getElementsByTagName("Duration").item(0).getTextContent());
                            List<Integer> dependencies = new ArrayList<>();
                            NodeList dependencyNodes = taskElement.getElementsByTagName("DependsOnTaskID");
                            for (int k = 0; k < dependencyNodes.getLength(); k++) {
                                dependencies.add(Integer.parseInt(dependencyNodes.item(k).getTextContent()));
                            }

                            taskList.add(new Task(taskID, description, duration, dependencies));
                        }
                    }

                    projectList.add(new Project(projectName, taskList));
                }

                

            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return projectList;
    }
}
