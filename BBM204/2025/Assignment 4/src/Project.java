import java.io.Serializable;
import java.util.*;

public class Project implements Serializable {
    static final long serialVersionUID = 33L;
    private final String name;
    private final List<Task> tasks;

    public Project(String name, List<Task> tasks) {
        this.name = name;
        this.tasks = tasks;
    }

    /**
     * @return the total duration of the project in days
     */
    public int getProjectDuration() {
        if (tasks.isEmpty()) return 0; // If no tasks, duration is zero.
    
        // Assume getEarliestSchedule method has been correctly implemented
        int[] startTimes = getEarliestSchedule();
        int projectDuration = 0;
    
        for (int i = 0; i < tasks.size(); i++) {
            int taskEnd = startTimes[i] + tasks.get(i).getDuration();
            if (taskEnd > projectDuration) {
                projectDuration = taskEnd;
            }
        }
    
        return projectDuration;
    }
    

    /**
     * Schedule all tasks within this project such that they will be completed as early as possible.
     *
     * @return An integer array consisting of the earliest start days for each task.
     */
    public int[] getEarliestSchedule() {
        int numTasks = tasks.size();
        int[] earliestStarts = new int[numTasks];
        int[] indegree = new int[numTasks];
        List<List<Integer>> adjList = new ArrayList<>();
    
        // Initialize adjacency list
        for (int i = 0; i < numTasks; i++) {
            adjList.add(new ArrayList<>());
        }
    
        // Build graph and compute indegrees
        for (int i = 0; i < numTasks; i++) {
            Task currentTask = tasks.get(i);
            for (Integer dep : currentTask.getDependencies()) {
                adjList.get(dep).add(i);
                indegree[i]++;
            }
        }
    
        // Queue for tasks that are ready to be processed
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numTasks; i++) {
            if (indegree[i] == 0) {
                queue.add(i);
                earliestStarts[i] = 0; // Start as soon as possible
            }
        }
    
        // Process tasks in topological order
        while (!queue.isEmpty()) {
            int current = queue.poll();
            for (int neighbor : adjList.get(current)) {
                indegree[neighbor]--;
                if (indegree[neighbor] == 0) {
                    queue.add(neighbor);
                }
                int possibleStart = earliestStarts[current] + tasks.get(current).getDuration();
                if (possibleStart > earliestStarts[neighbor]) {
                    earliestStarts[neighbor] = possibleStart;
                }
            }
        }
    
        return earliestStarts;
    }
    

    public static void printlnDash(int limit, char symbol) {
        for (int i = 0; i < limit; i++) System.out.print(symbol);
        System.out.println();
    }

    /**
     * Some free code here. YAAAY! 
     */
    public void printSchedule(int[] schedule) {
        int limit = 65;
        char symbol = '-';
        printlnDash(limit, symbol);
        System.out.println(String.format("Project name: %s", name));
        printlnDash(limit, symbol);

        // Print header
        System.out.println(String.format("%-10s%-45s%-7s%-5s","Task ID","Description","Start","End"));
        printlnDash(limit, symbol);
        for (int i = 0; i < schedule.length; i++) {
            Task t = tasks.get(i);
            System.out.println(String.format("%-10d%-45s%-7d%-5d", i, t.getDescription(), schedule[i], schedule[i]+t.getDuration()));
        }
        printlnDash(limit, symbol);
        System.out.println(String.format("Project will be completed in %d days.", tasks.get(schedule.length-1).getDuration() + schedule[schedule.length-1]));
        printlnDash(limit, symbol);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Project project = (Project) o;

        int equal = 0;

        for (Task otherTask : ((Project) o).tasks) {
            if (tasks.stream().anyMatch(t -> t.equals(otherTask))) {
                equal++;
            }
        }

        return name.equals(project.name) && equal == tasks.size();
    }

}
