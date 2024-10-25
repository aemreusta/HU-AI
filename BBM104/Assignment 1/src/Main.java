import java.io.IOException;

public class Main {

    public static void main(String[] args) throws IOException {
        
        /*
        String studentsFile = args[0];
        String coursesFile = args[1];
        String gradesFile = args[2];
        String commandsFile = args[3];
        String outputFile = args[4];
        */

        /*just for easy tries*/
        String studentsFile = "src/resource/students.txt";
        String coursesFile = "src/resource/courses.txt";
        String gradesFile = "src/resource/grades.txt";
        String commandsFile = "src/resource/commands.txt";
        String outputFile = "src/resource/outputs.txt";
        

        FileReader.studentReader(studentsFile);
        FileReader.courseReader(coursesFile);
        FileReader.gradesReader(gradesFile);
        Commands.readAndWrite(commandsFile, outputFile);

    }
    
}
