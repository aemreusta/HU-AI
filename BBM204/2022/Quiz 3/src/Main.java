import java.io.File;
import java.util.*;

public class Main {

    public static void main(String[] args) {


        int[][] maze = null;
        int[] sizes = new int[2];
        int[] goal = new int[2];

        try 
        {
            File myObj = new File("src/resources/map.txt");
            //File myObj = new File(args[0]);
            Scanner myReader = new Scanner(myObj);

            String strSize = myReader.nextLine();
            String[] strSizeArray = strSize.split(" ");
            sizes[0]= Integer.parseInt(strSizeArray[0]);
            sizes[1] = Integer.parseInt(strSizeArray[1]);

            maze = new int[sizes[0]][sizes[1]];
            
            for(int i = 0 ; i<sizes[0];i++)
            {
                String mazeRow = myReader.nextLine();
                String[] mazeRowArray = mazeRow.split(" ");
                int columnIndex = 0;
                
                for(String value : mazeRowArray)
                {
                    maze[i][columnIndex] = Integer.parseInt(value);
                    columnIndex +=1;
                }
            }

            String goalLine = myReader.nextLine();
            String[] goalArray = goalLine.split(" ");
            goal[0] = Integer.parseInt(goalArray[0]);
            goal[1] = Integer.parseInt(goalArray[1]);

            myReader.close();
        } 
        
        catch (Exception e) 
        {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }

        System.out.println("\nPossible Paths Number Between Robot and Target Tile: " + solve(maze,goal,sizes));
    }

    public static int solve(int[][] maze, int[] goal, int[] sizes) 
    {
        //if roads blocked
        if((maze[0][1] == 1 && maze[1][0] == 1) || maze[0][0] == 1 || maze[goal[0]][goal[1]] == 1)
            return 0;

        //create a second map for path finding as same size with the base map
        int[][] paths = new int[sizes[0]][sizes[1]];
        //set the starting position as 1 since it is reachable 
        paths[0][0] = 1;

        /*
        System.out.println("Maze Map\n-------");
        prtMap(maze);
        System.out.println("Target is 4,2\n\nPaths Maps After Every Row\n");
        */

        for(int i=0; i < sizes[0]; i++)
        {
            for(int j=0; j < sizes[1]; j++)
            {
                //if it is on the base position, continue
                if(i == 0 && j == 0)
                    continue;

                //if the next position is blocked, continue
                if(maze[i][j] == 1)
                    continue;

                if(i > 0)
                {
                    paths[i][j] += paths[i-1][j];
                }

                if(j > 0)
                {
                    paths[i][j] += paths[i][j-1];
                }

                //clean looking solution using '?' ternary operator
                /*
                paths[i][j] += i > 0 ? paths[i-1][j] : 0;
                paths[i][j] += j > 0 ? paths[i][j-1] : 0;
                */
            }

            //prtMap(paths);
        }

        //Return the result
        return paths[goal[0]][goal[1]];
    }

    //check the map to see if the method is progressing correctly
    public static void prtMap(int[][] map)
    {
        for(int i=0; i < map.length; i++)
        {
            for(int k=0; k < map[0].length; k++)
            {
                System.out.print(map[i][k] + " ");
            }
            System.out.println("");
        }
        System.out.println("-------");   
    }
}
