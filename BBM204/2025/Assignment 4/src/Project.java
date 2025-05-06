import java.io.Serializable;
import java.util.*;

public class Project implements Serializable {
    static final long serialVersionUID = 33L;
    private final String name;
    private final List<Task> tasks; 
    
    // Map TaskID to Task object for easier lookup
    private transient Map<Integer, Task> taskMap;
    // Adjacency list for topological sort (task ID -> list of task IDs that depend on it)
    private transient Map<Integer, List<Integer>> adj;
    // Cached earliest schedule and duration
    private transient int[] earliestScheduleCache = null;
    private transient Integer projectDurationCache = null;
    private transient List<Integer> topologicalOrderCache = null;


    public Project(String name, List<Task> tasks) {
        this.name = name;
        this.tasks = tasks;
        buildGraph(); // Build graph representation on creation
    }

    // Helper to build internal graph representation and task map
    private void buildGraph() {
        taskMap = new HashMap<>();
        adj = new HashMap<>();
        int numTasks = tasks.size();
        for (int i = 0; i < numTasks; i++) {
            Task t = tasks.get(i);
            if(t.getTaskID() != i) {
                 System.err.println("Warning: Task ID " + t.getTaskID() + " does not match index " + i + ". Scheduling might be incorrect if IDs aren't contiguous from 0.");
            }
            taskMap.put(t.getTaskID(), t);
            adj.put(t.getTaskID(), new ArrayList<>());
        }

        for (Task t : tasks) {
            for (int depId : t.getDependencies()) {
                 // adj stores: prerequisite -> tasks that depend on it
                if (adj.containsKey(depId)) {
                    adj.get(depId).add(t.getTaskID());
                } else {
                     System.err.println("Warning: Dependency Task ID " + depId + " not found for Task " + t.getTaskID());
                }
            }
        }
    }

     // Recalculate graph if deserialized
    private void readObject(java.io.ObjectInputStream in) throws java.io.IOException, ClassNotFoundException {
        in.defaultReadObject();
        buildGraph(); // Rebuild transient fields
    }


    /**
     * @return the total duration of the project in days
     */
    public int getProjectDuration() {
        if (projectDurationCache != null) {
            return projectDurationCache;
        }
        // Ensure schedule is calculated
        getEarliestSchedule();

        int projectDuration = 0;
        if (earliestScheduleCache != null && !tasks.isEmpty()) {
            for (int i = 0; i < tasks.size(); i++) {
                 Task t = taskMap.get(i); // Use taskMap for safety if IDs != indices
                 if (t != null) {
                     int endTime = earliestScheduleCache[i] + t.getDuration();
                     projectDuration = Math.max(projectDuration, endTime);
                 }
            }
        }
        projectDurationCache = projectDuration;
        return projectDuration;
    }

    /**
     * Schedule all tasks within this project such that they will be completed as early as possible.
     * Uses Kahn's algorithm (based on in-degrees) for topological sort and scheduling.
     *
     * @return An integer array consisting of the earliest start days for each task, ordered by Task ID (0 to n-1).
     */
    public int[] getEarliestSchedule() {
        if (earliestScheduleCache != null) {
            return earliestScheduleCache;
        }
        if (tasks.isEmpty()) {
            earliestScheduleCache = new int[0];
            topologicalOrderCache = new ArrayList<>();
            return earliestScheduleCache;
        }

        int numTasks = tasks.size();
        // Check if taskMap or adj are null (could happen after deserialization if readObject isn't called/implemented)
        if (taskMap == null || adj == null) {
            buildGraph();
        }

        int[] earliestStart = new int[numTasks];
        int[] inDegree = new int[numTasks];
        Arrays.fill(earliestStart, 0);

        // Calculate in-degrees
        for (Task t : tasks) {
            inDegree[t.getTaskID()] = t.getDependencies().size();
        }

        Queue<Integer> queue = new LinkedList<>();
        // Add all tasks with in-degree 0 to the queue
        for (int i = 0; i < numTasks; i++) {
            if (inDegree[i] == 0) {
                queue.offer(i);
            }
        }

        List<Integer> topologicalOrder = new ArrayList<>();
        while (!queue.isEmpty()) {
            int u = queue.poll();
            topologicalOrder.add(u);
            Task currentTask = taskMap.get(u);
            if (currentTask == null) continue; // Should not happen with buildGraph

            int earliestFinish = earliestStart[u] + currentTask.getDuration();

            // For all tasks v that depend on u
            if (adj.containsKey(u)) {
                for (int v : adj.get(u)) {
                    // Update earliest start of v if path through u is longer
                    earliestStart[v] = Math.max(earliestStart[v], earliestFinish);
                    inDegree[v]--;
                    if (inDegree[v] == 0) {
                        queue.offer(v);
                    }
                }
            }
        }

        if (topologicalOrder.size() != numTasks) {
            System.err.println("Error: Project '" + name + "' contains a cycle!");
            // Return partially calculated schedule or handle error appropriately
            // For now, cache what we have
             earliestScheduleCache = earliestStart;
             topologicalOrderCache = topologicalOrder; // Will be incomplete
             return earliestScheduleCache; // Or throw exception
        }

        earliestScheduleCache = earliestStart;
        topologicalOrderCache = topologicalOrder;
        return earliestScheduleCache;
    }

    public static void printlnDash(int limit, char symbol) {
        for (int i = 0; i < limit; i++) System.out.print(symbol);
        System.out.println();
    }

    /**
     * Prints the schedule in topological order.
     */
    public void printSchedule(int[] schedule) {
         // Ensure schedule and topological order are calculated
        if (earliestScheduleCache == null || topologicalOrderCache == null) {
            getEarliestSchedule();
        }
         // Handle case where schedule calculation failed (e.g., cycle)
        if (earliestScheduleCache == null || topologicalOrderCache == null) {
             System.err.println("Cannot print schedule for project '" + name + "' due to calculation errors.");
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

        // Print tasks in topological order
        for (int taskId : topologicalOrderCache) {
            Task t = taskMap.get(taskId);
             if (t != null) { // Check if task exists
                 // Use schedule array indexed by task ID
                 int startTime = schedule[taskId];
                 int endTime = startTime + t.getDuration();
                 System.out.println(String.format("%-10d%-45s%-7d%-5d", t.getTaskID(), t.getDescription(), startTime, endTime));
            }
        }
        printlnDash(limit, symbol);
        // Use the calculated project duration
        System.out.println(String.format("Project will be completed in %d days.", getProjectDuration()));
        printlnDash(limit, symbol);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Project project = (Project) o;

        // Simple equality check based on name and tasks list content might be sufficient
        // The original check was a bit complex; using standard list equality check
        return Objects.equals(name, project.name) &&
               Objects.equals(tasks, project.tasks); // Assumes Task equals() is correct
    }

     @Override
    public int hashCode() {
        return Objects.hash(name, tasks);
    }

}