# Shrimp Problem

The Shrimp Problem is a well-known optimization problem in which we have to find the minimum cost of purchasing different types of shrimps to clean a lake of a certain amount of algae in a given number of days.

This repository contains a Java solution to the Shrimp Problem. The code takes three inputs: the total grams of algae in the lake, the number of days available for cleaning the lake, and the number of different types of shrimps available. For each type of shrimp, the code takes in three integers: the cost of one unit of shrimp, the amount of algae one unit of shrimp can eat per day, and the total amount of that type of shrimp available.

The code then computes the optimal number of each type of shrimp to purchase to clean the lake in the given number of days. It does this by sorting the types of shrimp in descending order of their "efficiency", defined as the amount of algae one unit of shrimp can eat per unit cost. The code then purchases as many of each type of shrimp as necessary to eat the required amount of algae per day, up to the total amount of that type of shrimp available.

If there are not enough shrimps available to eat the required amount of algae per day, the code returns "Infeasible". Otherwise, it returns the total cost of the shrimps purchased to clean the lake.

## How to Use

To use this code, simply copy and paste the code into a Java IDE or text editor, compile it, and run it. When prompted, enter the total grams of algae in the lake, the number of days available for cleaning the lake, and the number of different types of shrimps available. Then, for each type of shrimp, enter the cost of one unit of shrimp, the amount of algae one unit of shrimp can eat per day, and the total amount of that type of shrimp available.

The code will then output the optimal number of each type of shrimp to purchase to clean the lake in the given number of days, along with the total cost of the shrimps purchased. If there are not enough shrimps available to eat the required amount of algae per day, the code will return "Infeasible".
