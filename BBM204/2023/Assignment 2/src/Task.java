import java.time.LocalTime;

public class Task implements Comparable {
    public String name;
    public String start;
    public int duration;
    public int importance;
    public boolean urgent;

    /*
     * Getter methods
     */
    public String getName() {
        return this.name;
    }

    public String getStartTime() {
        return this.start;
    }

    public int getDuration() {
        return this.duration;
    }

    public int getImportance() {
        return this.importance;
    }

    public boolean isUrgent() {
        return this.urgent;
    }

    /**
     * Finish time should be calculated here
     *
     * @return calculated finish time as String
     */
    public String getFinishTime() {
        // YOUR CODE HERE
        LocalTime finishTime = LocalTime.parse(start).plusHours(duration);
        return finishTime.toString();
    }

    /**
     * Weight calculation should be performed here
     *
     * @return calculated weight
     */
    public double getWeight() {
        // YOUR CODE HERE
        return (importance * (urgent ? 2000 : 1)) / duration;
    }

    /**
     * This method is needed to use {@link java.util.Arrays#sort(Object[])} ()},
     * which sorts the given array easily
     *
     * @param o Object to compare to
     * @return If self > object, return > 0 (e.g. 1)
     *         If self == object, return 0
     *         If self < object, return < 0 (e.g. -1)
     */
    @Override
    public int compareTo(Object o) {
        // YOUR CODE HERE
        // compare them using getFinishTime() method
        if (this.getFinishTime().compareTo(((Task) o).getFinishTime()) > 0) {
            return 1;
        }

        else if (this.getFinishTime().compareTo(((Task) o).getFinishTime()) == 0) {
            return 0;
        }

        else {
            return -1;
        }
    }
}
