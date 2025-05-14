import java.io.Serializable;
import java.util.*;

public class Project implements Serializable {
    static final long serialVersionUID = 33L;
    private final String name;
    private final List<Task> tasks;

    public Project(String name, List<Task> tasks) {
        this.name = name;
        this.tasks = new ArrayList<>(tasks); 
    }

    public String getName() {
        return name;
    }

    public List<Task> getTasks() {
        return tasks;
    }
    
    public List<Integer> getTopologicalOrder() {
        if (tasks.isEmpty()) {
            return new ArrayList<>();
        }

        int maxTaskId = -1;
        for (Task t : tasks) {
            if (t.getTaskID() > maxTaskId) {
                maxTaskId = t.getTaskID();
            }
        }
        if (maxTaskId == -1 && !tasks.isEmpty()) { // e.g. one task with ID 0
             maxTaskId = 0;
             for(Task t: tasks) if (t.getTaskID() > maxTaskId) maxTaskId = t.getTaskID();
        } else if (tasks.isEmpty()) {
             return new ArrayList<>();
        }


        int numTasksEffective = maxTaskId + 1;

        List<List<Integer>> adj = new ArrayList<>(numTasksEffective);
        int[] inDegree = new int[numTasksEffective];

        for (int i = 0; i < numTasksEffective; i++) {
            adj.add(new ArrayList<>());
        }
        
        Map<Integer, Task> taskMap = new HashMap<>();
        for (Task t : tasks) {
            taskMap.put(t.getTaskID(), t);
        }

        for (Task task : tasks) {
            int u = task.getTaskID();
            if (u < 0 || u >= numTasksEffective) continue; // Should not happen with correct maxTaskId logic
            for (int depID : task.getDependencies()) {
                if (depID < 0 || depID >= numTasksEffective) continue; // Dependency ID out of bounds
                if (taskMap.containsKey(depID)) { // Ensure dependency task actually exists
                    adj.get(depID).add(u);
                    inDegree[u]++;
                }
            }
        }

        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numTasksEffective; i++) {
            if (taskMap.containsKey(i) && inDegree[i] == 0) {
                queue.add(i);
            }
        }
        
        List<Integer> topologicalOrder = new ArrayList<>();
        while (!queue.isEmpty()) {
            int u = queue.poll();
            topologicalOrder.add(u);

            if (u < 0 || u >= adj.size()) continue; // Safety check

            for (int v : adj.get(u)) {
                if (v < 0 || v >= numTasksEffective) continue; // Successor ID out of bounds
                inDegree[v]--;
                if (inDegree[v] == 0 && taskMap.containsKey(v)) { // Ensure successor task exists
                    queue.add(v);
                }
            }
        }
        
        if (topologicalOrder.size() != this.tasks.size()) {
            // System.err.println("Warning: Topological sort size " + topologicalOrder.size() + " mismatch with tasks size " + this.tasks.size() + " for project " + this.name);
        }
        return topologicalOrder;
    }

    public int getProjectDuration() {
        if (tasks.isEmpty()) {
            return 0;
        }
        int[] schedule = getEarliestSchedule();
        if (schedule == null || schedule.length == 0) return 0; // getEarliestSchedule might return null/empty on error or empty tasks
        
        int projectDuration = 0;
        Map<Integer, Task> taskMap = new HashMap<>();
        for(Task t : tasks) {
            taskMap.put(t.getTaskID(), t);
        }

        for (Task task : tasks) {
            int taskID = task.getTaskID();
            if (taskID < 0 || taskID >= schedule.length) {
                // System.err.println("TaskID out of bounds for schedule array: " + taskID + " in project " + this.name);
                continue;
            }
            int endTime = schedule[taskID] + task.getDuration();
            if (endTime > projectDuration) {
                projectDuration = endTime;
            }
        }
        return projectDuration;
    }

    public int[] getEarliestSchedule() {
        if (tasks.isEmpty()) {
            return new int[0];
        }

        int maxTaskId = -1;
        for (Task t : tasks) {
            if (t.getTaskID() > maxTaskId) {
                maxTaskId = t.getTaskID();
            }
        }
        if (maxTaskId == -1 && !tasks.isEmpty()) { 
             maxTaskId = 0;
             for(Task t: tasks) if (t.getTaskID() > maxTaskId) maxTaskId = t.getTaskID();
        } else if (tasks.isEmpty()){
            return new int[0];
        }


        int numTasksEffective = maxTaskId + 1;
        int[] earliestStartTime = new int[numTasksEffective]; 
        
        List<List<Integer>> adj = new ArrayList<>(numTasksEffective); 
        int[] inDegree = new int[numTasksEffective];

        for (int i = 0; i < numTasksEffective; i++) {
            adj.add(new ArrayList<>());
        }
        
        Map<Integer, Task> taskMap = new HashMap<>();
        for(Task t : tasks) {
            taskMap.put(t.getTaskID(), t);
        }

        for (Task task : tasks) {
            int u = task.getTaskID();
             if (u < 0 || u >= numTasksEffective) continue;
            for (int depID : task.getDependencies()) {
                if (depID < 0 || depID >= numTasksEffective) continue; 
                if(taskMap.containsKey(depID)){ // Ensure dependency task exists
                    adj.get(depID).add(u);
                    inDegree[u]++;
                }
            }
        }

        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numTasksEffective; i++) {
            if (taskMap.containsKey(i) && inDegree[i] == 0) {
                queue.add(i);
                earliestStartTime[i] = 0; 
            }
        }
        
        while (!queue.isEmpty()) {
            int u = queue.poll();
            
            Task task_u = taskMap.get(u);
            if (task_u == null) continue; 
            int finish_time_u = earliestStartTime[u] + task_u.getDuration();

            if (u < 0 || u >= adj.size()) continue;

            for (int v : adj.get(u)) { 
                if (v < 0 || v >= numTasksEffective) continue;
                if(taskMap.containsKey(v)){ // Ensure successor task exists
                    earliestStartTime[v] = Math.max(earliestStartTime[v], finish_time_u);
                    inDegree[v]--;
                    if (inDegree[v] == 0) {
                        queue.add(v);
                    }
                }
            }
        }
        return earliestStartTime;
    }

    public static void printlnDash(int limit, char symbol) {
        for (int i = 0; i < limit; i++) System.out.print(symbol);
        System.out.println();
    }

    public void printSchedule(int[] schedule) {
        int limit = 65;
        char symbol = '-';
        printlnDash(limit, symbol);
        System.out.println(String.format("Project name: %s", name));
        printlnDash(limit, symbol);

        System.out.println(String.format("%-10s%-45s%-7s%-5s","Task ID","Description","Start","End"));
        printlnDash(limit, symbol);
        
        List<Integer> topoOrder = getTopologicalOrder(); 
        Map<Integer, Task> taskMap = new HashMap<>();
        for(Task currentTask : tasks){
            taskMap.put(currentTask.getTaskID(), currentTask);
        }

        for (int taskId : topoOrder) {
            Task t = taskMap.get(taskId);
            if (t != null && taskId < schedule.length && taskId >= 0) {
                System.out.println(String.format("%-10d%-45s%-7d%-5d", t.getTaskID(), t.getDescription(), schedule[t.getTaskID()], schedule[t.getTaskID()]+t.getDuration()));
            }
        }
        printlnDash(limit, symbol);
        System.out.println(String.format("Project will be completed in %d days.", getProjectDuration()));
        printlnDash(limit, symbol);
    }

    // Using the originally provided equals method
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Project project = (Project) o;

        if (!name.equals(project.name)) return false;
        // The original equals check task counts carefully
        if (this.tasks.size() != project.tasks.size()) return false;

        for (Task otherTask : project.tasks) {
            boolean found = false;
            for (Task thisTask : this.tasks) {
                if (thisTask.equals(otherTask)) {
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
        return true; // All tasks matched
    }

    // A hashCode method consistent with the above equals method
    @Override
    public int hashCode() {
        List<Task> sortedTasks = new ArrayList<>(this.tasks);
        sortedTasks.sort(Comparator.comparingInt(Task::getTaskID)); // Task.equals must be consistent
        
        int result = Objects.hash(name);
        result = 31 * result + sortedTasks.hashCode(); // Uses List.hashCode() on sorted list
        return result;
    }
}