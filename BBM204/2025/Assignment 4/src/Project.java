// Project.java
import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;


public class Project implements Serializable {
    static final long serialVersionUID = 33L;
    private final String name;
    private final List<Task> tasks; // Keep original order for printing
    private final Map<Integer, Task> taskMap; // For easy lookup by ID


    public Project(String name, List<Task> tasks) {
        this.name = name;
        // Ensure tasks are sorted by ID for consistent processing if not already
        this.tasks = tasks.stream()
                          .sorted(Comparator.comparingInt(Task::getTaskID))
                          .collect(Collectors.toList());
        this.taskMap = new HashMap<>();
        for (Task t : this.tasks) {
            this.taskMap.put(t.getTaskID(), t);
        }
    }

    // Getter for tasks if needed externally, maintaining original order
    public List<Task> getTasks() {
        return Collections.unmodifiableList(tasks);
    }

    /**
     * @return the total duration of the project in days
     */
    public int getProjectDuration() {
        int projectDuration = 0;
        int[] schedule = getEarliestSchedule();
        if (schedule == null || tasks.isEmpty()) {
            return 0; // No tasks or error in scheduling
        }

        for (Task task : tasks) {
            int taskIndex = task.getTaskID(); // Assumes taskID maps directly to index 0..N-1
             if (taskIndex >= schedule.length) {
                 // This might happen if TaskIDs are not contiguous from 0
                 // Or if getEarliestSchedule returned a smaller array due to error
                 System.err.println("Warning: Task ID " + taskIndex + " out of bounds for schedule array in project " + name);
                 continue;
             }
            int endTime = schedule[taskIndex] + task.getDuration();
            projectDuration = Math.max(projectDuration, endTime);
        }
        return projectDuration;
    }

    /**
     * Schedule all tasks within this project such that they will be completed as early as possible.
     * Uses Kahn's algorithm for topological sort and scheduling.
     *
     * @return An integer array consisting of the earliest start days for each task, indexed by Task ID.
     *         Returns null if a cycle is detected or tasks are invalid.
     */
    public int[] getEarliestSchedule() {
        if (tasks.isEmpty()) {
            return new int[0];
        }

        int n = tasks.size();
        // Find max Task ID to determine array size, assuming IDs might not be 0 to n-1 contiguously
         int maxTaskId = -1;
         for(Task t : tasks) {
             maxTaskId = Math.max(maxTaskId, t.getTaskID());
             for (int depId : t.getDependencies()) {
                 maxTaskId = Math.max(maxTaskId, depId);
             }
         }

         if (maxTaskId < 0 && n > 0) { // Handle case where there's one task with ID 0
            maxTaskId = 0;
         } else if (maxTaskId < 0) { // No tasks
             return new int[0];
         }


        int arraySize = maxTaskId + 1;
        int[] earliestStart = new int[arraySize]; // Indexed by TaskID
        int[] earliestFinish = new int[arraySize]; // Indexed by TaskID
        int[] inDegree = new int[arraySize];
        Map<Integer, List<Integer>> adj = new HashMap<>(); // Map: dependency -> list of tasks that depend on it

        // Initialize adjacency list and calculate in-degrees
        for (int i = 0; i < arraySize; i++) {
            adj.put(i, new ArrayList<>());
        }

        for (Task task : tasks) {
            int taskId = task.getTaskID();
            inDegree[taskId] = task.getDependencies().size();
            for (int depId : task.getDependencies()) {
                 if (!adj.containsKey(depId)) {
                    adj.put(depId, new ArrayList<>()); // Ensure dependency exists in map
                 }
                adj.get(depId).add(taskId);
            }
        }

        Queue<Integer> queue = new LinkedList<>();
        // Add tasks with no dependencies (in-degree 0) to the queue
        for (Task task : tasks) {
             int taskId = task.getTaskID();
            if (inDegree[taskId] == 0) {
                queue.offer(taskId);
                earliestStart[taskId] = 0; // Starts at day 0
                earliestFinish[taskId] = task.getDuration();
            }
        }

        int count = 0; // Count of processed tasks for cycle detection
        List<Integer> topologicalOrder = new ArrayList<>(); // Store the order for printing

        while (!queue.isEmpty()) {
            int u = queue.poll();
            topologicalOrder.add(u);
            count++;

            // If task u doesn't exist in the map (e.g., it's only a dependency ID), skip its processing
             Task taskU = taskMap.get(u);
             if (taskU == null) continue; // Should not happen if input is valid

             int finishTimeU = earliestStart[u] + taskU.getDuration();

            // For each task v that depends on u
            if (adj.containsKey(u)) {
                 for (int v : adj.get(u)) {
                     // Update earliest start time of v
                     earliestStart[v] = Math.max(earliestStart[v], finishTimeU);
                     inDegree[v]--;

                     // If in-degree becomes 0, add v to the queue
                     if (inDegree[v] == 0) {
                         Task taskV = taskMap.get(v);
                         if (taskV != null) { // Make sure it's an actual task
                            earliestFinish[v] = earliestStart[v] + taskV.getDuration();
                            queue.offer(v);
                         } else {
                             System.err.println("Warning: Task ID " + v + " found as dependency but not defined as a task in project " + name);
                         }
                     }
                 }
            }
        }

        // Check for cycles
        if (count != tasks.size()) {
            System.err.println("Error: Cycle detected in project '" + name + "'. Cannot determine schedule.");
            return null; // Indicate error
        }

        // Store the topological order for printing
        this.tasks.sort(Comparator.comparingInt(t -> topologicalOrder.indexOf(t.getTaskID())));

        // Return the schedule array indexed by TaskID
        return earliestStart;
    }

    public static void printlnDash(int limit, char symbol) {
        for (int i = 0; i < limit; i++) System.out.print(symbol);
        System.out.println();
    }

    /**
     * Prints the schedule based on the calculated earliest start times.
     * Tasks are printed in topological order determined by getEarliestSchedule.
     */
    public void printSchedule(int[] schedule) {
         if (schedule == null) {
             System.out.println("Cannot print schedule due to errors (e.g., cycle detected).");
             return;
         }
        int limit = 65;
        char symbol = '-';
        printlnDash(limit, symbol);
        System.out.println(String.format("Project name: %s", name));
        printlnDash(limit, symbol);

        // Print header
        System.out.println(String.format("%-10s%-45s%-7s%-5s","Task ID","Description","Start","End"));
        printlnDash(limit, symbol);

        // Print tasks in the topologically sorted order stored in this.tasks
        for (Task t : tasks) {
            int taskId = t.getTaskID();
            if (taskId >= schedule.length) {
                 System.err.println("Error printing schedule: Task ID " + taskId + " not found in schedule array.");
                 continue;
            }
            System.out.println(String.format("%-10d%-45s%-7d%-5d", taskId, t.getDescription(), schedule[taskId], schedule[taskId]+t.getDuration()));
        }

        printlnDash(limit, symbol);
        System.out.println(String.format("Project will be completed in %d days.", getProjectDuration()));
        printlnDash(limit, symbol);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Project project = (Project) o;
        // Use Objects.equals for name and compare task lists (order might matter depending on use case)
        // The original implementation checked if all tasks from 'o' exist in 'this', which isn't symmetric.
        // A better check might involve sets or comparing sorted lists if order doesn't matter.
        // For now, sticking closer to original logic but using taskMap for efficiency.
        if (!Objects.equals(name, project.name) || tasks.size() != project.tasks.size()) {
            return false;
        }
        // Check if all tasks are equal (assuming task.equals is robust)
        for (Task otherTask : project.tasks) {
            Task thisTask = taskMap.get(otherTask.getTaskID());
            if (thisTask == null || !thisTask.equals(otherTask)) {
                return false;
            }
        }
        return true;
    }

     @Override
     public int hashCode() {
         // Consistent with the modified equals
         return Objects.hash(name, taskMap);
     }
}