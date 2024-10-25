# Summer Assignment 1

Subject: Object Oriented Design, Classes and Objects

## Problem Definition

In this assignment, program takes 5 command lines argument as 5 different file addresses. 3 of these files contains information to setup a Student Information System. One of the other 2 contains commands to run the program. Other one name of the output file.

## Solution

To read a text file for setup information system, I created the FileReader class which has 3 different methods. Method related with students, read information, and create the student object. Then add it into an arraylist. About courses, do the same thing, create the object add it course arraylist.  Objects stored in these arraylists. The method about the grades file, read the info, find the course and the student objects in the arraylist, and add information inside the object as a HashMap. So, grades and course information store both course and student objects. This gives the simplicity to objects related methods.

For the read commands and write the results into file, I created the “Commands” class. Doing both reading and writing operations in the same class gives the algorithm simplicity and speed. Trickiest part of that part, reading the commands for me. Not using the correct database made it difficult to deal with the results of the commands. A faster program can be obtained with a more suitable database.
