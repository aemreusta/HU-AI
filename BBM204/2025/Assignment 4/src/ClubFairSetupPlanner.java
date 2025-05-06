import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Comparator; // <--- ADD THIS LINE

public class ClubFairSetupPlanner implements Serializable {
    static final long serialVersionUID = 88L;

    /**
     * Given a list of Project objects, prints the schedule of each of them.
     * Uses getEarliestSchedule() and printSchedule() methods of the current project to print its schedule.
     * @param projectList a list of Project objects
     */
    public void printSchedule(List<Project> projectList) {
        // TODO: YOUR CODE HERE - DONE
        if (projectList == null) {
            System.err.println("Project list is null. Cannot print schedule.");
            return;
        }
        for (Project project : projectList) {
            if (project != null) {
                int[] schedule = project.getEarliestSchedule(); // Calculate schedule
                 if (schedule != null) { // Check if schedule calculation was successful
                     project.printSchedule(schedule); // Print the schedule
                 } else {
                      // Use a getter for the name if it's private, or make it accessible
                      // System.err.println("Failed to calculate schedule for project: " + project.name);
                      // Assuming Project has a getName() method or name is public/package-private
                       System.err.println("Failed to calculate schedule for project: " + project.toString()); // Or a specific getName()
                 }
            }
        }
    }

    /**
     * Parse the input XML file and return a list of Project objects
     *
     * @param filename the input XML file
     * @return a list of Project objects, or an empty list if errors occur
     */
    public List<Project> readXML(String filename) {
        List<Project> projectList = new ArrayList<>();
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();

        try {
            DocumentBuilder builder = factory.newDocumentBuilder();
            Document document = builder.parse(new File(filename));
            document.getDocumentElement().normalize(); // Recommended practice

            NodeList projectNodes = document.getElementsByTagName("Project");

            for (int i = 0; i < projectNodes.getLength(); i++) {
                Node projectNode = projectNodes.item(i);
                if (projectNode.getNodeType() == Node.ELEMENT_NODE) {
                    Element projectElement = (Element) projectNode;

                    // Get Project Name
                    String projectName = projectElement.getElementsByTagName("Name").item(0).getTextContent();

                    // Get Tasks for this Project
                    List<Task> tasks = new ArrayList<>();
                    NodeList taskNodes = projectElement.getElementsByTagName("Task");

                    for (int j = 0; j < taskNodes.getLength(); j++) {
                        Node taskNode = taskNodes.item(j);
                        if (taskNode.getNodeType() == Node.ELEMENT_NODE) {
                            Element taskElement = (Element) taskNode;

                            // Extract Task details
                            int taskID = Integer.parseInt(taskElement.getElementsByTagName("TaskID").item(0).getTextContent());
                            String description = taskElement.getElementsByTagName("Description").item(0).getTextContent();
                            int duration = Integer.parseInt(taskElement.getElementsByTagName("Duration").item(0).getTextContent());

                            // Extract Dependencies
                            List<Integer> dependencies = new ArrayList<>();
                            NodeList dependenciesNodes = taskElement.getElementsByTagName("Dependencies");
                            if (dependenciesNodes.getLength() > 0) {
                                Node dependenciesNode = dependenciesNodes.item(0); // Get the <Dependencies> element itself
                                if (dependenciesNode != null && dependenciesNode.getNodeType() == Node.ELEMENT_NODE) {
                                     Element dependenciesElement = (Element) dependenciesNode;
                                     NodeList dependsOnNodes = dependenciesElement.getElementsByTagName("DependsOnTaskID");
                                     for (int k = 0; k < dependsOnNodes.getLength(); k++) {
                                         dependencies.add(Integer.parseInt(dependsOnNodes.item(k).getTextContent()));
                                     }
                                }
                            }
                            // Sort dependencies for consistent comparison if needed later
                            Collections.sort(dependencies);

                            // Create and add Task object
                            tasks.add(new Task(taskID, description, duration, dependencies));
                        }
                    }
                     // Sort tasks by ID to ensure list index matches TaskID (important for Project class logic)
                     tasks.sort(Comparator.comparingInt(Task::getTaskID)); // This line needs Comparator

                    // Create and add Project object
                    projectList.add(new Project(projectName, tasks));
                }
            }

        } catch (ParserConfigurationException | SAXException | IOException | NumberFormatException e) {
            System.err.println("Error parsing XML file '" + filename + "': " + e.getMessage());
            e.printStackTrace(); // Print stack trace for debugging
            return new ArrayList<>(); // Return empty list on error
        } catch (Exception e) { // Catch any other unexpected errors
             System.err.println("An unexpected error occurred during XML parsing: " + e.getMessage());
             e.printStackTrace();
             return new ArrayList<>();
        }

        return projectList;
    }
}