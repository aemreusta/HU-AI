import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class FileReader {
    
    public static ArrayList<Course> courseData = new ArrayList<>();
    public static ArrayList<Student> studentList = new ArrayList<>();

    public static void studentReader(String fileAddresses) throws FileNotFoundException
    {
        Scanner file = new Scanner(new File(fileAddresses));
        while(file.hasNextLine())
        {
            Integer studentId = file.nextInt();
            String studentName = file.next();
            String studentSurname = file.next();
            Student tmp_student = new Student(studentId, studentName, studentSurname);
            studentList.add(tmp_student);
        }
        file.close();
    }

    public static void courseReader(String fileAddresses) throws FileNotFoundException   
    {     
        Scanner file = new Scanner(new File(fileAddresses));
        while(file.hasNextLine())
        {
            Integer courseID = file.nextInt();
            String courseName = file.next();
            Course tmp_course = new Course(courseID, courseName);
            courseData.add(tmp_course);
        }
        file.close();
    }

    public static void gradesReader(String fileAddresses) throws FileNotFoundException   
    {     
        Scanner file = new Scanner(new File(fileAddresses));
        while(file.hasNextLine())
        {   
            String[] line = file.nextLine().split(" ");
            int[] intLine = new int[line.length];
            for(int k=0; k<line.length; k++)
            {   
                intLine[k] = Integer.parseInt(line[k]);
            }

            for(Course tmpCourse : courseData)
            {
                if(tmpCourse.iD == intLine[0])
                {
                    Integer tmpId = intLine[1];
                    int[] tmpScoreArray = Arrays.copyOfRange(intLine, 2, intLine.length);
                    tmpCourse.courseStudents.put(tmpId, tmpScoreArray);
                    break;
                }
            }

            for(Student tmpStudent : studentList)
            {
                if(tmpStudent.iD == intLine[1])
                {
                    Integer tmpId = intLine[0];
                    int[] tmpScoreArray = Arrays.copyOfRange(intLine, 2, intLine.length);
                    tmpStudent.coursesAndNotes.put(tmpId, tmpScoreArray);
                    break;
                }
            }
        }
        file.close();
    }           
}