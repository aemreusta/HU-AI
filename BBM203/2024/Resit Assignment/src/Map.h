#ifndef MAP_H
#define MAP_H

#include <iostream>
#include <memory>
#include <vector>
#include <fstream>
#include <queue>
#include <cmath>
#include "Isle.h"

struct MapNode
{
    Isle *isle;
    MapNode *left, *right;
    bool isRed; // Color of the node: red = true, black = false

    // Constructor
    MapNode(Isle *isle) : isle(isle), left(nullptr), right(nullptr), isRed(true) {}
    
    // Destructor
    ~MapNode()
    {
        // Freed in Map::~Map() using BFS traversal
    }
};

class Map
{
private:
    MapNode *root; // Root node of the tree

    // LLRBT helper method
    bool isRed(MapNode *node);

    // LLRBT rotations
    MapNode *rotateRight(MapNode *current);
    MapNode *rotateLeft(MapNode *current);

    // Recursive LLRBT insertion
    MapNode *insert(MapNode *node, Isle *isle);

    // Item distribution functions
    void preOrderItemDrop(MapNode *current, int &count);
    void postOrderItemDrop(MapNode *current, int &count);

    // Display helper
    void display(MapNode *current, int depth, int state);

public:
    // Constructor / Destructor
    Map();
    ~Map();

    MapNode* getRoot(); // Root getter

    // Tree operations
    void insert(Isle *isle);

    // Search
    Isle *findIsle(Isle isle);
    Isle *findIsle(std::string name);
    MapNode *findNode(Isle isle);
    MapNode *findNode(std::string name);

    // Initialize tree from a vector
    void initializeMap(std::vector<Isle *> isles);

    // Find first node with no item
    MapNode *findFirstEmptyIsle(MapNode *node);

    // Determines the depth of a node within the tree
    int getDepth(MapNode *node);

    // Display tree in terminal
    void displayMap();

    // Item drop
    void populateWithItems();
    void dropItemBFS();

    // Write tree (level order) to file
    void writeToFile(const std::string &filename);

    // Write current Isles (alphabetical order) to file
    void writeIslesToFile(const std::string &filename);
};

#endif
