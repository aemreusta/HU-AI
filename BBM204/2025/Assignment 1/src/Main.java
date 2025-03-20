import org.knowm.xchart.*;
import org.knowm.xchart.style.Styler;
import org.knowm.xchart.style.lines.SeriesLines;
import org.knowm.xchart.style.markers.SeriesMarkers;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;

public class Main {

    private static final int[] inputSizes = {500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 250000};
    private static final String filePath = "/Users/emre/GitHub/HU-AI/BBM204/2025/Assignment 1/src/resources/Traffic Flow Dataset.csv";  // Update file path here

    public static void main(String[] args) throws IOException {
        System.out.println("Starting main program execution...");

        ArrayList<int[]> arrays = readArraysFromFile(filePath, inputSizes);
        System.out.println("Arrays have been read from the file.");

        // Instantiate sorting classes
        InsertionSort insertionSort = new InsertionSort();
        CombSort combSort = new CombSort();
        RadixSort radixSort = new RadixSort();
        ShellSort shellSort = new ShellSort();
        ShakerSort shakerSort = new ShakerSort();
        System.out.println("Sorting classes have been instantiated.");

        // Experiment 1: Sorting on random data
        double[][] sortingTimesRandom = measureSortingPerformance(arrays, insertionSort, combSort, radixSort, shellSort, shakerSort);
        System.out.println("Sorting performance on random data has been measured.");
        
        // Experiment 2: Sorting on sorted data
        double[][] sortingTimesSorted = measureSortingPerformance(arrays, insertionSort, combSort, radixSort, shellSort, shakerSort);
        System.out.println("Sorting performance on sorted data has been measured.");
        
        // Experiment 3: Sorting on reversely sorted data
        reverseArrays(arrays);
        System.out.println("Arrays have been reversed for the experiment.");
        double[][] sortingTimesReversed = measureSortingPerformance(arrays, insertionSort, combSort, radixSort, shellSort, shakerSort);
        System.out.println("Sorting performance on reversed data has been measured.");

        // Save results to files
        saveResultsToFile("Sorting Performance on Random Data", inputSizes, sortingTimesRandom, new String[]{"Insertion Sort", "Comb Sort", "Radix Sort", "Shell Sort", "Shaker Sort"});
        System.out.println("Sorting performance on random data has been saved to file.");
        
        saveResultsToFile("Sorting Performance on Sorted Data", inputSizes, sortingTimesSorted, new String[]{"Insertion Sort", "Comb Sort", "Radix Sort", "Shell Sort", "Shaker Sort"});
        System.out.println("Sorting performance on sorted data has been saved to file.");
        
        saveResultsToFile("Sorting Performance on Reversed Data", inputSizes, sortingTimesReversed, new String[]{"Insertion Sort", "Comb Sort", "Radix Sort", "Shell Sort", "Shaker Sort"});
        System.out.println("Sorting performance on reversed data has been saved to file.");

        // Plotting results for sorting experiments
        plotResults("Sorting Performance on Random Data", inputSizes, sortingTimesRandom, new String[]{"Insertion Sort", "Comb Sort", "Radix Sort", "Shell Sort", "Shaker Sort"});
        System.out.println("Results for random data have been plotted.");
        
        plotResults("Sorting Performance on Sorted Data", inputSizes, sortingTimesSorted, new String[]{"Insertion Sort", "Comb Sort", "Radix Sort", "Shell Sort", "Shaker Sort"});
        System.out.println("Results for sorted data have been plotted.");
        
        plotResults("Sorting Performance on Reversed Data", inputSizes, sortingTimesReversed, new String[]{"Insertion Sort", "Comb Sort", "Radix Sort", "Shell Sort", "Shaker Sort"});
        System.out.println("Results for reversed data have been plotted.");

        System.out.println("Program execution completed.");
    }

    // Reads arrays from the file based on specified input sizes
    public static ArrayList<int[]> readArraysFromFile(String filePath, int[] inputSizes) throws IOException {
        System.out.println("Reading arrays from file: " + filePath);

        ArrayList<int[]> arrays = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        
        String line;
        ArrayList<Integer> allNumbers = new ArrayList<>();
        
        // Skip header line
        reader.readLine();
        
        // Read "Flow Duration" values into a list
        while ((line = reader.readLine()) != null) {
            try {
                String[] parts = line.split(",");
                int flowDuration = Integer.parseInt(parts[2]);
                allNumbers.add(flowDuration);
            } catch (NumberFormatException e) {
                System.err.println("Skipping invalid or incorrectly formatted line: " + line);
            }
        }
        reader.close();
        
        // Create arrays for each input size
        for (int size : inputSizes) {
            if (allNumbers.size() >= size) {
                int[] array = new int[size];
                for (int i = 0; i < size; i++) {
                    array[i] = allNumbers.get(i);
                }
                arrays.add(array);
            } else {
                System.err.println("Not enough data for requested input size: " + size);
            }
        }
        
        System.out.println("Arrays have been successfully read from the file.");
        return arrays;
    }

    // Measures sorting performance for each algorithm
    private static double[][] measureSortingPerformance(ArrayList<int[]> arrays, InsertionSort insertionSort, CombSort combSort, RadixSort radixSort, ShellSort shellSort, ShakerSort shakerSort) {
        System.out.println("Measuring sorting performance...");

        double[][] times = new double[5][inputSizes.length];
        for (int i = 0; i < inputSizes.length; i++) {
            int[] originalArray = arrays.get(i);

            for (int trial = 0; trial < 10; trial++) {
                int[] arrayToSort;

                // Measure Insertion Sort
                arrayToSort = Arrays.copyOf(originalArray, originalArray.length);
                long startTime = System.nanoTime();
                insertionSort.sort(arrayToSort);
                long endTime = System.nanoTime();
                times[0][i] += (endTime - startTime) / 1e6;

                // Measure Comb Sort
                arrayToSort = Arrays.copyOf(originalArray, originalArray.length);
                startTime = System.nanoTime();
                combSort.sort(arrayToSort);
                endTime = System.nanoTime();
                times[1][i] += (endTime - startTime) / 1e6;

                // Measure Radix Sort
                arrayToSort = Arrays.copyOf(originalArray, originalArray.length);
                startTime = System.nanoTime();
                radixSort.sort(arrayToSort);
                endTime = System.nanoTime();
                times[2][i] += (endTime - startTime) / 1e6;

                // Measure Shell Sort
                arrayToSort = Arrays.copyOf(originalArray, originalArray.length);
                startTime = System.nanoTime();
                shellSort.sort(arrayToSort);
                endTime = System.nanoTime();
                times[3][i] += (endTime - startTime) / 1e6;

                // Measure Shaker Sort
                arrayToSort = Arrays.copyOf(originalArray, originalArray.length);
                startTime = System.nanoTime();
                shakerSort.sort(arrayToSort);
                endTime = System.nanoTime();
                times[4][i] += (endTime - startTime) / 1e6;
            }

            // Average times
            for (int j = 0; j < 5; j++) {
                times[j][i] /= 10;
            }
        }
        System.out.println("Sorting performance measurement completed.");
        return times;
    }

    // Reverses arrays for reverse sorting experiment
    private static void reverseArrays(ArrayList<int[]> arrays) {
        System.out.println("Reversing arrays...");

        for (int[] array : arrays) {
            for (int i = 0; i < array.length / 2; i++) {
                int temp = array[i];
                array[i] = array[array.length - i - 1];
                array[array.length - i - 1] = temp;
            }
        }
        System.out.println("Arrays have been successfully reversed.");
    }

    // Plots sorting results with XChart
    private static void plotResults(String title, int[] inputSizes, double[][] yAxis, String[] seriesNames) {
        System.out.println("Plotting results for: " + title);

        double[] xData = Arrays.stream(inputSizes).mapToDouble(i -> i).toArray();
        XYChart chart = new XYChartBuilder().width(800).height(600).title(title).xAxisTitle("Input Size").yAxisTitle("Time (ms)").build();

        chart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNE);
        chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line);
        chart.getStyler().setMarkerSize(6);

        for (int i = 0; i < yAxis.length; i++) {
            XYSeries series = chart.addSeries(seriesNames[i], xData, yAxis[i]);
            series.setMarker(SeriesMarkers.CIRCLE);
            series.setLineStyle(SeriesLines.SOLID);
        }

        new SwingWrapper<>(chart).displayChart();

        // Save chart as image
        File graphFolder = new File("./graphs");
        if (!graphFolder.exists()) {
            graphFolder.mkdirs();
        }
        String fileName = title.replace(" ", "_") + ".png";
        try {
            BitmapEncoder.saveBitmap(chart, "./graphs/" + fileName, BitmapEncoder.BitmapFormat.PNG);
            System.out.println("Saved graph to: ./graphs/" + fileName);
        } catch (IOException e) {
            System.err.println("Error saving graph: " + e.getMessage());
        }
    }

    // Saves sorting performance results to a file
    private static void saveResultsToFile(String title, int[] inputSizes, double[][] yAxis, String[] seriesNames) {
        System.out.println("Saving results to file: " + title);

        File resultsFolder = new File("./results");
        if (!resultsFolder.exists()) {
            resultsFolder.mkdirs();
        }

        String fileName = title.replace(" ", "_") + ".txt";
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File(resultsFolder, fileName)))) {
            writer.write(title + "\n");
            writer.write("Input Size");
            for (String seriesName : seriesNames) {
                writer.write("\t" + seriesName);
            }
            writer.write("\n");

            for (int i = 0; i < inputSizes.length; i++) {
                writer.write(inputSizes[i] + "");
                for (double[] timeSeries : yAxis) {
                    writer.write("\t" + timeSeries[i]);
                }
                writer.write("\n");
            }

            System.out.println("Saved results to: ./results/" + fileName);
        } catch (IOException e) {
            System.err.println("Error writing results to file: " + e.getMessage());
        }
    }
}
