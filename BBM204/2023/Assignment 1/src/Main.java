import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;

import org.knowm.xchart.*;
import org.knowm.xchart.style.Styler;

class Main {
    public static void main(String[] args) throws IOException {

        // Read subsets from CSV file
        int[] sizes = { 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 250000 };

        Integer[][] randomSubsets = CSVReader.readSubsets("/resources/TrafficFlowDataset.csv", sizes);

        // Sort every subset using Arrays.sort()
        Integer[][] sortSubsets = new Integer[10][];
        for (int i = 0; i < sortSubsets.length; i++) {
            sortSubsets[i] = Arrays.copyOf(randomSubsets[i], randomSubsets[i].length);
            Arrays.sort(sortSubsets[i]);
        }

        // Reverse sorted subsets using Arrays.sort()
        Integer[][] reverseSubsets = new Integer[10][];
        for (int i = 0; i < reverseSubsets.length; i++) {
            reverseSubsets[i] = Arrays.copyOf(sortSubsets[i], sortSubsets[i].length);
            Arrays.sort(reverseSubsets[i], Collections.reverseOrder());
        }

        // Sort every subset using SelectionSort
        Double[] selectionTimes = Utils.sortingRun(randomSubsets);
        System.out.println(Arrays.toString(selectionTimes));

        // X axis data
        int[] inputAxis = { 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 251282 };

        // Create sample data for linear runtime
        double[][] yAxis = new double[2][10];
        yAxis[0] = new double[] { 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 251282 };
        yAxis[1] = new double[] { 300, 800, 1800, 3000, 7000, 15000, 31000, 64000, 121000, 231000 };

        // Save the char as .png and show it
        showAndSaveChart("Sample Test", inputAxis, yAxis);
    }

    public static void showAndSaveChart(String title, int[] xAxis, double[][] yAxis) throws IOException {
        // Create Chart
        XYChart chart = new XYChartBuilder().width(800).height(600).title(title)
                .yAxisTitle("Time in Milliseconds").xAxisTitle("Input Size").build();

        // Convert x axis to double[]
        double[] doubleX = Arrays.stream(xAxis).asDoubleStream().toArray();

        // Customize Chart
        chart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNE);
        chart.getStyler().setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line);

        // Add a plot for a sorting algorithm
        chart.addSeries("Sample Data 1", doubleX, yAxis[0]);
        chart.addSeries("Sample Data 2", doubleX, yAxis[1]);

        // Save the chart as PNG
        BitmapEncoder.saveBitmap(chart, title + ".png", BitmapEncoder.BitmapFormat.PNG);

        // Show the chart
        new SwingWrapper(chart).displayChart();
    }
}
