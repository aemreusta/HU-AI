import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) throws IOException {

        /** Safe-lock Opening Algorithm Below **/

        System.out.println("##Initiate Operation Safe-lock##");
        // TODO: Your code goes here
        // You are expected to read the file given as the first command-line argument to read
        // safes arriving each minute. Then, use this data to instantiate a
        // OptimalScrollSolution object. You need to call optimalSafeOpeningAlgorithm() method
        // of your MaxScrollsDP object to get the solution, and finally print it using
        // printSolution() method of OptimalScrollSolution object.
        System.out.println("##Operation Safe-lock Completed##");

        /** Operation Artifact Algorithm Below **/

        System.out.println("##Initiate Operation Artifact##");
        // TODO: Your code goes here
        // You are expected to read the file given as the second command-line argument to read
        // each artifact. Then, use this data to instantiate an OptimalShipSolution object.
        // You need to call optimalArtifactCarryingAlgorithm() method
        // of your MinShipsGP object to get the solution, and finally print it using
        // printSolution() method of OptimalShipSolution object.
        System.out.println("##Operation Artifact Completed##");

    }
}