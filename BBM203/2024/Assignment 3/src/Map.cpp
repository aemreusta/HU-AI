#include "Map.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

Map::Map() {
    distanceMatrix.resize(MAX_SIZE, std::vector<int>(MAX_SIZE, -1)); // Initialize all distances to -1 (no direct connection)
    for (int i = 0; i < MAX_SIZE; ++i) {
        visited[i] = false; // Initialize all provinces as unvisited
    }
}

// Loads distance data from a file and fills the distanceMatrix
void Map::loadDistanceData(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    std::string line;
    int row = 0;
    while (std::getline(file, line) && row < MAX_SIZE) {
        std::istringstream ss(line);
        std::string cell;
        int col = 0;
        while (std::getline(ss, cell, ',') && col < MAX_SIZE) {
            distanceMatrix[row][col] = std::stoi(cell);
            col++;
        }
        row++;
    }
    file.close();
}

// Checks if the distance between two provinces is within the allowed maxDistance
bool Map::isWithinRange(int provinceA, int provinceB, int maxDistance) const {
    if (provinceA < 0 || provinceA >= MAX_SIZE || provinceB < 0 || provinceB >= MAX_SIZE) {
        return false;
    }
    return distanceMatrix[provinceA][provinceB] != -1 && distanceMatrix[provinceA][provinceB] <= maxDistance;
}

// Marks a province as visited
void Map::markAsVisited(int province) {
    if (province >= 0 && province < MAX_SIZE) {
        visited[province] = true;
    }
}

// Checks if a province has already been visited
bool Map::isVisited(int province) const {
    if (province >= 0 && province < MAX_SIZE) {
        return visited[province];
    }
    return false;
}

// Resets all provinces to unvisited
void Map::resetVisited() {
    for (int i = 0; i < MAX_SIZE; ++i) {
        visited[i] = false;
    }
}

// Function to count the number of visited provinces
int Map::countVisitedProvinces() const {
    int count = 0;
    for (int i = 0; i < MAX_SIZE; ++i) {
        if (visited[i]) {
            count++;
        }
    }
    return count;
}

// Function to get the distance between two provinces
int Map::getDistance(int provinceA, int provinceB) const {
    if (provinceA < 0 || provinceA >= MAX_SIZE || provinceB < 0 || provinceB >= MAX_SIZE) {
        return -1; // Invalid province index
    }
    return distanceMatrix[provinceA][provinceB];
}