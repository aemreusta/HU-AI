import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.DocumentBuilder;
import org.w3c.dom.Document;
import org.w3c.dom.NodeList;
import org.w3c.dom.Node;
import org.w3c.dom.Element;
import java.io.File; 
import java.io.FileInputStream; 
import java.io.InputStream; 

import java.io.Serializable;
import java.util.*;

public class ClubFairSetupPlanner implements Serializable {
    static final long serialVersionUID = 88L;

    public void printSchedule(List<Project> projectList) {
        for (Project project : projectList) {
            int[] schedule = project.getEarliestSchedule(); 
            
            int limit = 65;
            char symbol = '-';
            Project.printlnDash(limit, symbol); 
            System.out.println(String.format("Project name: %s", project.getName()));
            Project.printlnDash(limit, symbol);

            System.out.println(String.format("%-10s%-45s%-7s%-5s","Task ID","Description","Start","End"));
            Project.printlnDash(limit, symbol);

            List<Integer> topologicalOrderIds = project.getTopologicalOrder();
            
            Map<Integer, Task> taskMap = new HashMap<>();
            for(Task t : project.getTasks()){ // project.getTasks() returns the original list
                taskMap.put(t.getTaskID(), t);
            }

            for (int taskId : topologicalOrderIds) {
                Task t = taskMap.get(taskId);
                if (t != null && taskId < schedule.length && taskId >= 0) { 
                    System.out.println(String.format("%-10d%-45s%-7d%-5d",
                            t.getTaskID(),
                            t.getDescription(),
                            schedule[t.getTaskID()], 
                            schedule[t.getTaskID()] + t.getDuration()));
                }
            }
            Project.printlnDash(limit, symbol);
            System.out.println(String.format("Project will be completed in %d days.", project.getProjectDuration()));
            Project.printlnDash(limit, symbol);
        }
    }

    public List<Project> readXML(String filename) {
        List<Project> projectList = new ArrayList<>();
        try {
            // Using InputStream to be more flexible with how test env provides files
            InputStream inputFileStream = new FileInputStream(filename); 
            DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
            Document doc = dBuilder.parse(inputFileStream); // Parse from InputStream
            doc.getDocumentElement().normalize();
            inputFileStream.close(); // Close the stream

            NodeList projectNodes = doc.getElementsByTagName("Project");

            for (int i = 0; i < projectNodes.getLength(); i++) {
                Node projectNode = projectNodes.item(i);
                if (projectNode.getNodeType() == Node.ELEMENT_NODE) {
                    Element projectElement = (Element) projectNode;
                    String projectName = projectElement.getElementsByTagName("Name").item(0).getTextContent().trim();
                    
                    List<Task> tasks = new ArrayList<>();
                    NodeList taskNodes = projectElement.getElementsByTagName("Task");

                    for (int j = 0; j < taskNodes.getLength(); j++) {
                        Node taskNode = taskNodes.item(j);
                        if (taskNode.getNodeType() == Node.ELEMENT_NODE) {
                            Element taskElement = (Element) taskNode;
                            int taskID = Integer.parseInt(taskElement.getElementsByTagName("TaskID").item(0).getTextContent().trim());
                            String description = taskElement.getElementsByTagName("Description").item(0).getTextContent().trim();
                            int duration = Integer.parseInt(taskElement.getElementsByTagName("Duration").item(0).getTextContent().trim());
                            
                            List<Integer> dependencies = new ArrayList<>();
                            NodeList depNodesList = taskElement.getElementsByTagName("Dependencies"); // Get the <Dependencies> wrapper
                            if (depNodesList.getLength() > 0) {
                                Node dependenciesNode = depNodesList.item(0);
                                if (dependenciesNode != null && dependenciesNode.getNodeType() == Node.ELEMENT_NODE) {
                                    Element dependenciesElement = (Element) dependenciesNode;
                                    NodeList depTaskIDNodes = dependenciesElement.getElementsByTagName("DependsOnTaskID");
                                    for (int k = 0; k < depTaskIDNodes.getLength(); k++) {
                                        dependencies.add(Integer.parseInt(depTaskIDNodes.item(k).getTextContent().trim()));
                                    }
                                }
                            }
                            tasks.add(new Task(taskID, description, duration, dependencies));
                        }
                    }
                    // Sort tasks by ID to maintain the original order
                    projectList.add(new Project(projectName, tasks));
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return projectList;
    }
}