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
        // TODO: YOUR CODE HERE
        return projectDuration;
    }

    /**
     * Schedule all tasks within this project such that they will be completed as late as possible.
     *
     * @return An integer array consisting of the latest start days for each task.
     */
    public int[] getLatestSchedule(int projectDeadline) {
        // TODO: YOUR CODE HERE
        return null;
    }


    /**
     * Schedule all tasks within this project such that they will be completed as early as possible.
     *
     * @return An integer array consisting of the earliest start days for each task.
     */
    public int[] getEarliestSchedule() {
        // TODO: YOUR CODE HERE
        return null;
    }


    /**
     * Using both the earliest and the latest schedules, returns a list containing the mobility of each task.
     */
    public int [] getMobility(int [] earliestSchedule, int [] latestSchedule) {
        // TODO: YOUR CODE HERE
        int [] mobility = new int[0];
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
