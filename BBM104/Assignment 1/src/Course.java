import java.util.HashMap;

public class Course {
    
    public Integer iD;
    public String courseName;
    /*This hashmap contains individuals ID's who take the course as a key and
     *their exam notes as value of hashmap in a integer array.*/
     public HashMap<Integer, int[]> courseStudents = new HashMap<>(); 

    public Course(Integer iD, String courseName)
    {
        this.iD = iD;
        this.courseName = courseName;
    }

    @Override
    public String toString() {
        return(iD+" "+courseName);
    }

    public String bestStudent()
    {
        int bestPossibleId = 0;
        double bestScore = 0.0;

        for(Integer tmpId : courseStudents.keySet())
        {
            double bestPossible=0.0;
            int[] tmpScores = courseStudents.get(tmpId);

            if(tmpScores.length == 3)
            {
                bestPossible = tmpScores[0]*(0.25)+tmpScores[1]*(0.25)+tmpScores[2]*(0.5);
            }

            else if(tmpScores.length == 2)
            {
                bestPossible = tmpScores[0]*(0.4)+tmpScores[1]*(0.6);
            }
            
            if((bestPossible)>bestScore)
            {
                bestPossibleId = tmpId;
                bestScore = bestPossible;
            }
        }
        
        for(Student tmpStudent : FileReader.studentList)
        {
            if(tmpStudent.iD.equals(bestPossibleId))
            {
                //System.out.println(String.format(" %.2f",bestScore));
                return(tmpStudent.toString()+String.format(" %.2f",bestScore));
            }
        }

        return("");
    }

    public String courseAverage()
    {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        int divider = courseStudents.size();

        for(Integer tmpInteger : courseStudents.keySet())
        {
            int[] tmpScores = courseStudents.get(tmpInteger);

            if(tmpScores.length == 3)
            {
                x += tmpScores[0];
                y += tmpScores[1];
                z += tmpScores[2];
            }

            if(tmpScores.length == 2)
            {
                x += tmpScores[0];
                y += tmpScores[1];
            }
        }

        if(divider == 3)
        {
            return(String.format("%.2f %.2f %.2f",(x/divider),(y/divider),(z/divider)));
        }

        else
            return(String.format("%.2f %.2f",(x/divider),(y/divider)));   
        
    }
}
