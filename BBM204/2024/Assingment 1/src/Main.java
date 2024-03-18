import org.knowm.xchart.*;
import org.knowm.xchart.style.Styler;
import org.knowm.xchart.style.lines.SeriesLines;
import org.knowm.xchart.style.markers.SeriesMarkers;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class Main {

    private static final int[] inputSizes = {500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 250000};
    private static final String filePath = "/Users/emre/GitHub/HU-AI/BBM204/2024/Assingment 1/src/resources/TrafficFlowDataset.csv";

    public static void main(String[] args) throws IOException {
        ArrayList<int[]> arrays = readArraysFromFile(filePath, inputSizes);
    
        // Instantiate sorting and searching classes
        InsertionSort insertionSort = new InsertionSort();
        MergeSort mergeSort = new MergeSort();
        CountingSort countingSort = new CountingSort();
        LinearSearch linearSearch = new LinearSearch();
        BinarySearch binarySearch = new BinarySearch();
    
        // Experiment 1: Sorting on random data
        double[][] sortingTimesRandom = measureSortingPerformance(arrays, insertionSort, mergeSort, countingSort);
        
        // Experiment 2: Sorting on sorted data
        double[][] sortingTimesSorted = measureSortingPerformance(arrays, insertionSort, mergeSort, countingSort);
        
        // Experiment 3: Sorting on reversely sorted data
        reverseArrays(arrays);
        double[][] sortingTimesReversed = measureSortingPerformance(arrays, insertionSort, mergeSort, countingSort);
    
        // Experiment 4: Searching on random data
        // Note: For linear search on random data, use the original unsorted arrays before they were sorted in sorting experiments
        ArrayList<int[]> randomDataArrays = readArraysFromFile(filePath, inputSizes); // Re-read or deep copy the original data if necessary
        double[][] searchingTimesRandom = measureSearchingPerformance(randomDataArrays, linearSearch, binarySearch, false);
    
        // Experiment 5: Searching on sorted data
        // Use the sorted arrays from the sorting experiments for linear and binary search on sorted data
        double[][] searchingTimesSorted = measureSearchingPerformance(arrays, linearSearch, binarySearch, true);
    
        // Save results to files
        saveResultsToFile("Sorting Performance on Random Data", inputSizes, sortingTimesRandom, new String[]{"Insertion Sort", "Merge Sort", "Counting Sort"});
        saveResultsToFile("Sorting Performance on Sorted Data", inputSizes, sortingTimesSorted, new String[]{"Insertion Sort", "Merge Sort", "Counting Sort"});
        saveResultsToFile("Sorting Performance on Reversed Data", inputSizes, sortingTimesReversed, new String[]{"Insertion Sort", "Merge Sort", "Counting Sort"});


        // Plotting results for sorting experiments
        plotResults("Sorting Performance on Random Data", inputSizes, sortingTimesRandom, new String[]{"Insertion Sort", "Merge Sort", "Counting Sort"});
        plotResults("Sorting Performance on Sorted Data", inputSizes, sortingTimesSorted, new String[]{"Insertion Sort", "Merge Sort", "Counting Sort"});
        plotResults("Sorting Performance on Reversed Data", inputSizes, sortingTimesReversed, new String[]{"Insertion Sort", "Merge Sort", "Counting Sort"});

        // Save results to files
        saveResultsToFile("Searching Performance on Random Data", inputSizes, searchingTimesRandom, new String[]{"Linear Search", "Binary Search (N/A)"});
        saveResultsToFile("Searching Performance on Sorted Data", inputSizes, searchingTimesSorted, new String[]{"Linear Search", "Binary Search"});

        // Plotting results for searching experiments (adapt as needed for your plotting function)
        plotResults("Searching Performance on Random Data", inputSizes, searchingTimesRandom, new String[]{"Linear Search", "Binary Search (N/A)"});
        plotResults("Searching Performance on Sorted Data", inputSizes, searchingTimesSorted, new String[]{"Linear Search", "Binary Search"});
    }
    

    private static ArrayList<int[]> readArraysFromFile(String filePath, int[] inputSizes) throws IOException {
        ArrayList<int[]> arrays = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        
        // Assuming the file is structured with integers in each row or a specific column
        String line;
        ArrayList<Integer> allNumbers = new ArrayList<>();
        
        // Read all numbers into a list (adjust parsing as necessary for your data format)
        while ((line = reader.readLine()) != null) {
            try {
                // Adjust this parsing depending on the structure of your CSV
                String[] parts = line.split(","); // Assuming the number is in a specific column
                int number = Integer.parseInt(parts[parts.length - 1]); // Example: parsing the last column
                allNumbers.add(number);
            } catch (NumberFormatException e) {
                // Handle potential format issues, e.g., non-integer values or empty lines
                System.err.println("Skipping invalid or incorrectly formatted line: " + line);
            }
        }
        reader.close();
        
        // Create arrays for each input size using the first n rows from the dataset
        for (int size : inputSizes) {
            // Ensure there are enough numbers for the requested size
            if (allNumbers.size() >= size) {
                int[] array = new int[size];
                for (int i = 0; i < size; i++) {
                    array[i] = allNumbers.get(i);
                }
                arrays.add(array);
            } else {
                // Log an error or throw an exception if there aren't enough numbers for the requested size
                System.err.println("Not enough data in file for requested input size: " + size);
            }
        }
        
        return arrays;
    }
    
    private static double[][] measureSearchingPerformance(ArrayList<int[]> arrays, LinearSearch linearSearch, BinarySearch binarySearch, boolean isDataSorted) {
        Random random = new Random();
        // 2 search algorithms, number of input sizes
        double[][] times = new double[2][inputSizes.length];
    
        for (int i = 0; i < inputSizes.length; i++) {
            int[] array = arrays.get(i);
            // Ensure binary search is only run on sorted data
            if (isDataSorted) {
                Arrays.sort(array); // Ensure the array is sorted for binary search, redundant if already sorted
            }
            
            for (int j = 0; j < 1000; j++) {
                // Pick a random target number from the array for each trial
                int target = array[random.nextInt(array.length)];
    
                // Measure Linear Search
                long startTime = System.nanoTime();
                linearSearch.search(array, target);
                long endTime = System.nanoTime();
                times[0][i] += (endTime - startTime);
    
                if (isDataSorted) {
                    // Measure Binary Search if data is sorted
                    startTime = System.nanoTime();
                    binarySearch.search(array, target); // Assuming binarySearch.search() exists and is implemented correctly
                    endTime = System.nanoTime();
                    times[1][i] += (endTime - startTime);
                }
            }
    
            // Calculate average time in nanoseconds for each input size
            times[0][i] /= 1000; // Average for Linear Search
            if (isDataSorted) {
                times[1][i] /= 1000; // Average for Binary Search
            }
        }
        return times;
    }
    


    private static double[][] measureSortingPerformance(ArrayList<int[]> arrays, InsertionSort insertionSort, MergeSort mergeSort, CountingSort countingSort) {
    double[][] times = new double[3][inputSizes.length];
    for (int i = 0; i < inputSizes.length; i++) {
        // Get the array for the current input size
        int[] originalArray = arrays.get(i);

        for (int trial = 0; trial < 10; trial++) {
            // Deep copy the original array to avoid sorting already sorted arrays
            int[] arrayToSort = Arrays.copyOf(originalArray, originalArray.length);

            // Measure Insertion Sort
            long startTime = System.nanoTime();
            insertionSort.sort(arrayToSort);
            long endTime = System.nanoTime();
            times[0][i] += (endTime - startTime) / 1e6; // Convert to milliseconds

            // Reset array for next sort
            arrayToSort = Arrays.copyOf(originalArray, originalArray.length);

            // Measure Merge Sort
            startTime = System.nanoTime();
            mergeSort.sort(arrayToSort);
            endTime = System.nanoTime();
            times[1][i] += (endTime - startTime) / 1e6; // Convert to milliseconds

            // Reset array for next sort
            arrayToSort = Arrays.copyOf(originalArray, originalArray.length);

            // Measure Counting Sort
            startTime = System.nanoTime();
            countingSort.sort(arrayToSort);
            endTime = System.nanoTime();
            times[2][i] += (endTime - startTime) / 1e6; // Convert to milliseconds
        }

        // Calculate average times for each algorithm and input size
        times[0][i] /= 10; // Average for Insertion Sort
        times[1][i] /= 10; // Average for Merge Sort
        times[2][i] /= 10; // Average for Counting Sort
    }
    return times;
}

    private static void reverseArrays(ArrayList<int[]> arrays) {
        for (int[] array : arrays) {
            for(int i = 0; i < array.length / 2; i++) {
                int temp = array[i];
                array[i] = array[array.length - i - 1];
                array[array.length - i - 1] = temp;
            }
        }
    }

    private static void plotResults(String title, int[] inputSizes, double[][] yAxis, String[] seriesNames) {
    // Convert inputSizes to double array
    double[] xData = Arrays.stream(inputSizes).mapToDouble(i -> i).toArray();

    // Create a chart
    XYChart chart = new XYChartBuilder().width(800).height(600).title(title).xAxisTitle("Input Size").yAxisTitle("Time (ms)").build();

    // Customize chart
    chart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNE);
    chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line);
    chart.getStyler().setMarkerSize(6);

    // Add series for each sorting or searching algorithm
    for (int i = 0; i < yAxis.length; i++) {
        XYSeries series = chart.addSeries(seriesNames[i], xData, yAxis[i]);
        series.setMarker(SeriesMarkers.CIRCLE);
        series.setLineStyle(SeriesLines.SOLID);
    }

    // Display the chart
    new SwingWrapper<>(chart).displayChart();

    // Save the chart as an image
    String graphFolderPath = "./graphs"; // Relative path to the graphs folder from the project root
    File graphFolder = new File(graphFolderPath);
    if (!graphFolder.exists()) {
        graphFolder.mkdirs(); // Create the folder if it doesn't exist
    }
    String fileName = title.replace(" ", "_") + ".png"; // Create a file name from the chart title
    try {
        BitmapEncoder.saveBitmap(chart, graphFolderPath + "/" + fileName, BitmapEncoder.BitmapFormat.PNG);
        System.out.println("Saved graph to: " + graphFolderPath + "/" + fileName);
    } catch (IOException e) {
        System.err.println("Error saving graph: " + e.getMessage());
    }
    }

    private static void saveResultsToFile(String title, int[] inputSizes, double[][] yAxis, String[] seriesNames) {
    String resultsFolderPath = "./results"; // Relative path to the results folder from the project root
    File resultsFolder = new File(resultsFolderPath);
    if (!resultsFolder.exists()) {
        resultsFolder.mkdirs(); // Create the folder if it doesn't exist
    }

    String fileName = title.replace(" ", "_") + ".txt"; // Create a file name from the title
    File file = new File(resultsFolderPath + "/" + fileName);

    try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
        // Write the title and column headers
        writer.write(title + "\n");
        writer.write("Input Size");
        for (String seriesName : seriesNames) {
            writer.write("\t" + seriesName);
        }
        writer.write("\n");

        // Write the data for each input size
        for (int i = 0; i < inputSizes.length; i++) {
            writer.write(inputSizes[i] + "");
            for (double[] timeSeries : yAxis) {
                writer.write("\t" + timeSeries[i]);
            }
            writer.write("\n");
        }

        System.out.println("Saved results to: " + file.getAbsolutePath());
    } catch (IOException e) {
        System.err.println("Error writing results to file: " + e.getMessage());
    }
    }

}
