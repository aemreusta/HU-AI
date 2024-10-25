import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class CSVReader {
    public static Integer[][] readSubsets(String resourceName, int[] subsetSizes) {
        Integer[][] subsets = new Integer[subsetSizes.length][]; // 2D array of subsets

        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(CSVReader.class.getResourceAsStream(resourceName)))) {
            String line;
            String csvSplitBy = ",";
            int subsetIndex = 0;
            ArrayList<Integer> lastColumn = new ArrayList<>();
            br.readLine(); // skip first line

            while ((line = br.readLine()) != null && subsetIndex < subsetSizes.length) {
                String[] data = line.split(csvSplitBy);
                lastColumn.add(Integer.parseInt(data[6])); // last column

                if (lastColumn.size() == subsetSizes[subsetIndex]) {
                    subsets[subsetIndex] = lastColumn.toArray(new Integer[0]);
                    subsetIndex++;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return subsets;
    }
}
