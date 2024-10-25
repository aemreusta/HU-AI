import java.io.File;
import java.io.Serializable;
import java.util.*;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

public class UrbanInfrastructureDevelopment implements Serializable {
    static final long serialVersionUID = 88L;

    /**
     * Given a list of Project objects, prints the schedule of each of them.
     * Uses getEarliestSchedule() and printSchedule() methods of the current project to print its schedule.
     * @param projectList a list of Project objects
     */
    public void printSchedule(List<Project> projectList) {
        for (Project project : projectList) {
            int[] schedule = project.getEarliestSchedule();
            project.printSchedule(schedule);
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
        DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
        DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
        Document doc = dBuilder.parse(new File(filename));
        doc.getDocumentElement().normalize();

        NodeList nodeList = doc.getElementsByTagName("Project");

        for (int i = 0; i < nodeList.getLength(); i++) {
            Node node = nodeList.item(i);

            if (node.getNodeType() == Node.ELEMENT_NODE) {
                Element element = (Element) node;
                String projectName = element.getElementsByTagName("Name").item(0).getTextContent();
                List<Task> tasks = new ArrayList<>();

                NodeList tasksList = element.getElementsByTagName("Task");
                for (int j = 0; j < tasksList.getLength(); j++) {
                    Node taskNode = tasksList.item(j);
                    if (taskNode.getNodeType() == Node.ELEMENT_NODE) {
                        Element taskElement = (Element) taskNode;
                        int taskID = Integer.parseInt(taskElement.getElementsByTagName("TaskID").item(0).getTextContent());
                        String description = taskElement.getElementsByTagName("Description").item(0).getTextContent();
                        int duration = Integer.parseInt(taskElement.getElementsByTagName("Duration").item(0).getTextContent());
                        List<Integer> dependencies = new ArrayList<>();

                        NodeList dependenciesList = taskElement.getElementsByTagName("DependsOnTaskID");
                        for (int k = 0; k < dependenciesList.getLength(); k++) {
                            dependencies.add(Integer.parseInt(dependenciesList.item(k).getTextContent()));
                        }
                        tasks.add(new Task(taskID, description, duration, dependencies));
                    }
                }
                projectList.add(new Project(projectName, tasks));
            }
        }
    } catch (Exception e) {
        e.printStackTrace();
    }
    return projectList;
}
}
