import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class Commands {
    
    public static void readAndWrite(String commandsFile, String outputFileName) throws IOException 
    {
        FileWriter clerk = new FileWriter(new File(outputFileName));
        Scanner file = new Scanner(new File(commandsFile));

        while(file.hasNextLine())
        {
            String line = file.nextLine();
            String[] lineArray = line.split(" ");
            //System.out.println(line[0]);
            //System.out.println(Arrays.toString(line));
            clerk.write("COMMAND:\n");
            clerk.write(line+"\n");
            clerk.write("RESULT:\n");
            
            if(lineArray[0].equals("LIST"))
            {
                if(lineArray[1].equals("COURSES"))
                {
                    if(lineArray[2].equals("ALL"))
                    {
                        for(Course tmpCourse : FileReader.courseData)
                        {
                            clerk.write(tmpCourse.toString()+"\n");
                        }
                        clerk.write("\n");
                    }

                    else
                    {
                        for(Student tmpStudent : FileReader.studentList)
                        {
                            if(tmpStudent.iD.equals(Integer.parseInt(lineArray[4])))
                            {
                                for(Integer tmpId : tmpStudent.coursesAndNotes.keySet())
                                {
                                    for(Course tmpCourse : FileReader.courseData)
                                    {
                                        if(tmpCourse.iD.equals(tmpId))
                                        {
                                            clerk.write(tmpCourse.toString()+"\n");
                                        }            
                                    }
                                }
                            }
                        }
                        clerk.write("\n");
                    }
                }

                else if(lineArray[1].equals("STUDENTS"))
                {
                    for(Course tmpCourse : FileReader.courseData)
                    {
                        if(tmpCourse.iD.equals(Integer.parseInt(lineArray[4])))
                        {
                            for(Integer tmpId : tmpCourse.courseStudents.keySet())
                            {
                                for(Student tmpStudent : FileReader.studentList)
                                {
                                    if(tmpStudent.iD.equals(tmpId))
                                    {
                                        clerk.write(tmpStudent.toString()+"\n");
                                     }
                                 }
                             }
                        }
                    } 
                    clerk.write("\n");   
                }

                else if(lineArray[1].equals("GRADES"))
                {
                    if(lineArray[3].equals("STUDENT"))
                    {
                        for(Student tmpStudent : FileReader.studentList)
                        {
                            if(tmpStudent.iD.equals(Integer.parseInt(lineArray[4])))
                            {
                                clerk.write(tmpStudent.toStringHashMap()+"\n");
                            }
                        }
                    }

                    else if(lineArray[3].equals("COURSE"))
                    {   //This functions result of command "LIST GRADES FOR COURSE <courseId> AND STUDENT <studentId>"
                        for(Course tmpCourse : FileReader.courseData)
                        {
                            if(tmpCourse.iD.equals(Integer.parseInt(lineArray[4])))
                            {
                                for(int wrt : tmpCourse.courseStudents.get(Integer.parseInt(lineArray[7])))
                                {
                                    clerk.write(wrt+" ");
                                }
                            }
                        }
                        clerk.write("\n\n");
                    }
                }
            }

            else if(lineArray[0].equals("FIND"))
            {
                for(Course tmpCourse : FileReader.courseData)
                {
                    if(tmpCourse.iD.equals(Integer.parseInt(lineArray[4])))
                    {
                        clerk.write(tmpCourse.bestStudent());
                    }
                }
                clerk.write("\n\n");
            }

            else if(lineArray[0].equals("CALCULATE"))
            {
                if(lineArray[1].equals("FINALGRADE"))
                {   //This functions result of "CALCULATE FINALGRADE FOR COURSE <courseId> AND STUDENT <studentId>"
                    for(Student tmpStudent : FileReader.studentList)
                    {
                        if(tmpStudent.iD.equals(Integer.parseInt(lineArray[7])))
                        {
                            clerk.write(tmpStudent.calculateFinal(Integer.parseInt(lineArray[4]))+"\n\n");
                        }
                    }
                }

                else if(lineArray[1].equals("ALL"))
                {   //This functions result of "CALCULATE ALL FINALGRADES FOR STUDENT <studentId>"
                    for(Student tmpStudent : FileReader.studentList)
                    {
                        if(tmpStudent.iD.equals(Integer.parseInt(lineArray[5])))
                        {
                            for(Integer tmpCourseId : tmpStudent.coursesAndNotes.keySet())
                            {
                                clerk.write(tmpCourseId+" "+(tmpStudent.calculateFinal(tmpCourseId))+"\n");
                            }
                            clerk.write("\n");     
                        }
                    }
                }

                else if(lineArray[1].equals("AVERAGE"))
                {   //This functions result of "CALCULATE AVERAGE GRADES FOR COURSE <courseId>"
                    for(Course tmpCourse : FileReader.courseData)
                    {
                        if(tmpCourse.iD.equals(Integer.parseInt(lineArray[5])))
                        {
                            clerk.write(tmpCourse.courseAverage()+"\n\n");
                        }
                    }
                }
            }

        }
        file.close();
        clerk.close();
    }
}
