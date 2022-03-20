# Graded Quiz 3

*Subject: Dynamic Programming*

## Problem Definition

In this quiz, program takes the map and find the possible paths number between robot and target. '0' represents an open road, '1' obstacle. To better understanding of problem, you can check the PDF prepared from course instructors.

## Solution

Main goal of this assignment is finding a solution using dynamic programming approach. To do that, I create secondary map same size as first map and store the paths on it. If the road is open, tile increase recursively. At the end, we check the target tile number and find out the possible paths the that road.

------

## Output of Program
```
Maze Map
-------
0 0 0 
0 1 0
0 0 0
0 0 0
0 1 0
-------
Target is 4,2

Paths Maps After Every Row

1 1 1
0 0 0
0 0 0
0 0 0
0 0 0
-------
1 1 1
1 0 1
0 0 0
0 0 0
0 0 0
-------
1 1 1
1 0 1
1 1 2
0 0 0
0 0 0
-------
1 1 1
1 0 1
1 1 2
1 2 4 
0 0 0
-------
1 1 1
1 0 1
1 1 2
1 2 4
1 0 4
-------

Possible Paths Number Between Robot and Target Tile: 4
```