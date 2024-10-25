import java.io.*;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;

import com.google.gson.Gson;
import com.google.gson.stream.JsonReader;

public class Main {
    /**
     * Propogated {@link IOException} here
     * {@link #parseJSON} method should be called here
     * A {@link Planner} instance must be instantiated here
     */
    public static void main(String[] args) throws IOException {
        String file = "/Users/emre/GitHub/HU-AI/BBM204/2023/Assignment 2/io/input/input6.json";
        // String file = args[0]; // get file name as an argument
        Task[] tasks = parseJSON(file);
        Arrays.sort(tasks); // sort task array for binary search

        for (Task t : tasks) {
            System.out.println(t);
        }

        Planner planner = new Planner(tasks);
        // System.out.println("Dynamic Schedule\n---------------");
        // ArrayList<Task> tmp = planner.planDynamic();
        // for (Task t : tmp) {
        // System.out.println(t);
        // }

        System.out.println("Greedy Schedule\n---------------");
        ArrayList<Task> tmp2 = planner.planGreedy();
        for (Task t : tmp2) {
            System.out.println(t);
        }
    }

    /**
     * @param filename json filename to read
     * @return Returns a list of {@link Task}s obtained by reading the given json
     *         file
     * @throws FileNotFoundException If the given file does not exist
     */
    public static Task[] parseJSON(String filename) throws FileNotFoundException {

        /* JSON parsing operations here */
        Gson gson = new Gson();
        JsonReader jr = new JsonReader(new FileReader(filename));
        return gson.fromJson(jr, Task[].class);
    }
}
