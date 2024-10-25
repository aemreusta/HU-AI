import java.util.HashMap;

public class Student {
    
    public Integer iD;
    public String name;
    public String surname;
    /*This hashmap contains courses ID's as a key which taken from a student and
     *students' exam notes at this course as value of hashmap in a integer array.*/
    public HashMap<Integer, int[]> coursesAndNotes = new HashMap<>();

    public Student(Integer iD, String name, String surname) {
        this.iD = iD;
        this.name = name;
        this.surname = surname;
    }

    @Override
    public String toString() {
        return(iD+" "+name+" "+surname);
    }

    public String toStringHashMap(){
        String readyToWrite = "";
        for(Integer tmpId : coursesAndNotes.keySet())
        {
            readyToWrite += (tmpId+" ");
            for(int i=0; i<coursesAndNotes.get(tmpId).length; i++)
            {
                readyToWrite += (coursesAndNotes.get(tmpId)[i]+" ");
            }            
            readyToWrite += "\n";
        }
        return readyToWrite;
    }

    public String calculateFinal(Integer courseId)
    {
        double finalGrade = 0.0;
        int[] tmpScores = coursesAndNotes.get(courseId);
        
        if(tmpScores.length == 3)
        {
            finalGrade = tmpScores[0]*(0.25)+tmpScores[1]*(0.25)+tmpScores[2]*(0.5);
        }

        else if(tmpScores.length == 2)
        {
            finalGrade = tmpScores[0]*(0.4)+tmpScores[1]*(0.6);
        }

        return String.format("%.2f",finalGrade);
    }
}
