import java.io.Serializable;
import java.util.*;

public class Project implements Serializable {

    private static final long serialVersionUID = 1L;

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
        int projectDuration = 0;

        // call getEarliestSchedule() and return the last element of the array
        int[] earliestSchedule = getEarliestSchedule();
        projectDuration = earliestSchedule[earliestSchedule.length-1] + tasks.get(earliestSchedule.length-1).getDuration();

        return projectDuration;
    }
    

    /**
     * Schedule all tasks within this project such that they will be completed as late as possible.
     *
     * @param projectDeadline The deadline by which the project must be completed.
     * @return An integer array consisting of the latest start days for each task.
     */
    public int[] getLatestSchedule(int projectDeadline) {
        int numTasks = tasks.size();
        int[] latestStarts = new int[numTasks];
        boolean[] visited = new boolean[numTasks];
        List<List<Integer>> adjList = new ArrayList<>();

        // Initialize adjacency list
        for (int i = 0; i < numTasks; i++) {
            adjList.add(new ArrayList<>());
        }

        // Build graph and calculate outdegrees
        for (int i = 0; i < numTasks; i++) {
            Task currentTask = tasks.get(i);
            for (Integer successor : currentTask.getDependencies()) {
                adjList.get(successor).add(i); // Store successors for each task
            }
        }

        // Use a recursive function to compute latest start times starting from task 0
        computeLatestStarts(0, adjList, latestStarts, visited, projectDeadline);

        return latestStarts;
    }

    /**
     * Recursive function to compute latest start times.
     *
     * @param taskID         The current task ID being processed.
     * @param adjList        Adjacency list representing task dependencies.
     * @param latestStarts   Array to store the latest start times for each task.
     * @param visited        Array to track visited tasks.
     * @param projectDeadline The project deadline.
     */
    private void computeLatestStarts(int taskID, List<List<Integer>> adjList, int[] latestStarts, boolean[] visited, int projectDeadline) {
        if (visited[taskID]) {
            return;
        }

        visited[taskID] = true;

        // Recursively process dependencies first
        for (int successor : adjList.get(taskID)) {
            computeLatestStarts(successor, adjList, latestStarts, visited, projectDeadline);
        }

        // Calculate latest start time for current task
        int latestStart = projectDeadline - tasks.get(taskID).getDuration();
        for (int successor : adjList.get(taskID)) {
            latestStart = Math.min(latestStart, latestStarts[successor] - tasks.get(taskID).getDuration());
        }

        // Ensure task 0 starts at the beginning
        if (taskID == 0) {
            latestStart = 0;
        }

        latestStarts[taskID] = latestStart;
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


    /**
     * Using both the earliest and the latest schedules, returns a list containing the mobility of each task.
     */
    public int[] getMobility(int[] earliestSchedule, int[] latestSchedule) {
        int[] mobility = new int[tasks.size()];
        
        for (int i = 0; i < tasks.size(); i++) {
            mobility[i] = latestSchedule[i] - earliestSchedule[i];
        }
        
        return mobility;
    }
    


    public void printMobility(int [] mobility) {
        int limit = 65;
        char symbol = '-';

        printlnDash(limit, symbol);
        System.out.println("Mobility Analysis");

        printlnDash(limit, symbol);

        System.out.println(String.format("%-10s%-45s%-7s","Task ID","Description","Mobility"));
        printlnDash(limit, symbol);

        for (int i = 0; i < mobility.length; i++) {
            Task t = tasks.get(i);
            System.out.println(String.format("%-10d%-45s%-7d", i, t.getDescription(), mobility[i]));
        }

        printlnDash(limit, symbol);
        System.out.println("Critical Tasks");
        printlnDash(limit, symbol);

        System.out.println(String.format("%-10s%-45s","Task ID","Description"));
        printlnDash(limit, symbol);
        for (int i = 0; i < mobility.length; i++) {
            if (mobility[i] == 0) {
                Task t = tasks.get(i);
                System.out.println(String.format("%-10d%-45s", i, t.getDescription()));
            }
        }

        printlnDash(limit, symbol);
    }

    public static void printlnDash(int limit, char symbol) {
        for (int i = 0; i < limit; i++) System.out.print(symbol);
        System.out.println();
    }

    public void printSchedule(int[] schedule, String scheduleType) {
        int limit = 65;
        char symbol = '-';

        printlnDash(limit, symbol);
        System.out.println(String.format("Schedule Type: %s", scheduleType));
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
