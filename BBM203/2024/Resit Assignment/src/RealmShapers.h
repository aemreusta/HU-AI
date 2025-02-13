#ifndef MASTERS_H
#define MASTERS_H

#include <vector>
#include <iostream>
#include <string>
#include "RealmShaper.h"

// This tree has a vector/array based implementation
class ShaperTree
{
protected:
    // Array to implement the tree
    std::vector<RealmShaper *> realmShapers; // Might be possible to use array as well

    // Helper function for safe index checking
    bool isValidIndex(int index);

public:
    // Constructor declaration
    ShaperTree();
    // Destructor
    ~ShaperTree();

    // Getters
    int getSize();
    std::vector<RealmShaper *> getTree(); // If array is used instead of an vector this function and ONLY this function should be changed
    RealmShaper *getParent(RealmShaper *shaper);
    int getDepth(RealmShaper *shaper); // Determines the depth level of a shaper within the tree.
    int getDepth();                    // Total depth of the tree

    void initializeTree(std::vector<RealmShaper *> players); // Initilize tree from a vector

    // Tree operations
    void insert(RealmShaper *shaper);
    int remove(RealmShaper *shaper);

    /// Search
    int findIndex(RealmShaper *shaper);
    RealmShaper *findPlayer(std::string name);
    RealmShaper *findPlayer(RealmShaper shaper);

    // Traversal functions
    std::vector<std::string> inOrderTraversal(int index);
    std::vector<std::string> preOrderTraversal(int index);
    void preOrderHelper(int index, std::vector<std::string>& result);
    std::vector<std::string> postOrderTraversal(int index);
    void postOrderHelper(int index, std::vector<std::string>& result);
    void breadthFirstTraversal(std::ofstream &outFile);
    
    // Terminal and file outputs
    void displayTree();
    int getRightChildIndex(int index);
    int getLeftChildIndex(int index);
    void printTree(int index, int level, const std::string &prefix);
};

#endif