# Hash Table Implementation

This repository contains a Java program that implements a simple hash table. The program takes a series of positive integer values and inserts them at indices that are computed using a straightforward hash function. In case of a collision, the hashtable resolves it using the open-addressing approach.

## Usage

1. Install the Java Development Kit (JDK) on your system.
2. Clone the repository or download the source code and navigate to the project directory in your terminal or command prompt.
3. Compile the Java program using the following command:

```bash
javac HashTable.java
```

4. Run the program with an input file using the following command:

```bash
java HashTable < input.txt
```

Note: Replace input.txt with the name of the input file you want to use.

5. The program will output the string representation of the hashtable after all values are inserted.

## Input Format

The input file should contain the following:

1. The size of the hash table (N) on the first line.
2. A space-separated list of positive integer values to be inserted into the hash table on the second line.

## Output Format

The program outputs the string representation of the hashtable after all values are inserted. The hashtable is represented as a single-line string as follows:

```text
0:v0|1:v1|2:v2|...|N-1:vN-1|
```

Where `v0` to `vN-1` are the values inserted in the corresponding indices of the hashtable. If an index is unoccupied, the corresponding value is `0`.

## Example

Suppose we have the following input file (`input.txt`):

```text
41
41 82 123 164 205 246 287 328 384 240 116 167 350 272 173 196 384 101 240 197
```

We can run the program with this input file and get the following output:

```text
0:41|1:82|2:0|3:167|4:123|5:0|6:0|7:0|8:328|9:164|10:173|11:0|12:0|13:0|14:0|15:384|16:205|17:0|18:0|19:384|20:101|21:0|22:350|23:0|24:0|25:246|26:272|27:0|28:0|29:0|30:0|31:0|32:196|33:197|34:116|35:240|36:287|37:0|38:0|39:240|40:0|
```
