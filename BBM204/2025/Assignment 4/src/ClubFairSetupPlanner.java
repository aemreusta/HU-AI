// ClubFairSetupPlanner.java
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ClubFairSetupPlanner implements Serializable {
    static final long serialVersionUID = 88L;

    /**
     * Given a list of Project objects, prints the schedule of each of them.
     * Uses getEarliestSchedule() and printSchedule() methods of the current project to print its schedule.
     * @param projectList a list of Project objects
     */
    public void printSchedule(List<Project> projectList) {
        if (projectList == null || projectList.isEmpty()) {
            System.out.println("No projects to schedule.");
            return;
        }
        for (int i = 0; i < projectList.size(); i++) {
            Project project = projectList.get(i);
            int[] schedule = project.getEarliestSchedule();
            project.printSchedule(schedule);
             // Print separator only between projects
            // if (i < projectList.size() - 1) {
            //    Project.printlnDash(65, '-'); // Use the dash printer from Project
            //}
        }
    }

    /**
     * Parses the input XML file and returns a list of Project objects.
     *
     * @param filename the input XML file
     * @return a list of Project objects, or null on error
     */
    public List<Project> readXML(String filename) {
        List<Project> projectList = new ArrayList<>();
        try {
            File xmlFile = new File(filename);
            DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
            Document doc = dBuilder.parse(xmlFile);
            doc.getDocumentElement().normalize();

            NodeList projectNodes = doc.getElementsByTagName("Project");

            for (int i = 0; i < projectNodes.getLength(); i++) {
                Node projectNode = projectNodes.item(i);
                if (projectNode.getNodeType() == Node.ELEMENT_NODE) {
                    Element projectElement = (Element) projectNode;
                    String projectName = projectElement.getElementsByTagName("Name").item(0).getTextContent();
                    List<Task> tasks = new ArrayList<>();

                    NodeList taskNodes = projectElement.getElementsByTagName("Task");
                    for (int j = 0; j < taskNodes.getLength(); j++) {
                        Node taskNode = taskNodes.item(j);
                        if (taskNode.getNodeType() == Node.ELEMENT_NODE) {
                            Element taskElement = (Element) taskNode;
                            int taskId = Integer.parseInt(taskElement.getElementsByTagName("TaskID").item(0).getTextContent());
                            String description = taskElement.getElementsByTagName("Description").item(0).getTextContent();
                            int duration = Integer.parseInt(taskElement.getElementsByTagName("Duration").item(0).getTextContent());

                            List<Integer> dependencies = new ArrayList<>();
                            NodeList depNodes = ((Element) taskElement.getElementsByTagName("Dependencies").item(0)).getElementsByTagName("DependsOnTaskID");
                            for (int k = 0; k < depNodes.getLength(); k++) {
                                dependencies.add(Integer.parseInt(depNodes.item(k).getTextContent()));
                            }
                            Collections.sort(dependencies); // Keep dependencies sorted if needed

                            tasks.add(new Task(taskId, description, duration, dependencies));
                        }
                    }
                     // Sort tasks by ID before creating the project for consistency
                    // tasks.sort(Comparator.comparingInt(Task::getTaskID));
                    projectList.add(new Project(projectName, tasks));
                }
            }
        } catch (ParserConfigurationException | SAXException | IOException | NumberFormatException | NullPointerException e) {
            System.err.println("Error reading or parsing XML file: " + filename);
            e.printStackTrace();
            return null; // Return null to indicate failure
        }
        return projectList;
    }
}