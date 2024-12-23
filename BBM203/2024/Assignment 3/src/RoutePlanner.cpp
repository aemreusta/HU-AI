#include "RoutePlanner.h"
#include <iostream>
#include <fstream>
#include <sstream>

// Array to help you out with name of the cities in order
const std::string cities[81] = { 
    "Adana", "Adiyaman", "Afyon", "Agri", "Amasya", "Ankara", "Antalya", "Artvin", "Aydin", "Balikesir", "Bilecik", 
    "Bingol", "Bitlis", "Bolu", "Burdur", "Bursa", "Canakkale", "Cankiri", "Corum", "Denizli", "Diyarbakir", "Edirne", 
    "Elazig", "Erzincan", "Erzurum", "Eskisehir", "Gaziantep", "Giresun", "Gumushane", "Hakkari", "Hatay", "Isparta", 
    "Mersin", "Istanbul", "Izmir", "Kars", "Kastamonu", "Kayseri", "Kirklareli", "Kirsehir", "Kocaeli", "Konya", "Kutahya", 
    "Malatya", "Manisa", "Kaharamanmaras", "Mardin", "Mugla", "Mus", "Nevsehir", "Nigde", "Ordu", "Rize", "Sakarya", 
    "Samsun", "Siirt", "Sinop", "Sivas", "Tekirdag", "Tokat", "Trabzon", "Tunceli", "Urfa", "Usak", "Van", "Yozgat", 
    "Zonguldak", "Aksaray", "Bayburt", "Karaman", "Kirikkale", "Batman", "Sirnak", "Bartin", "Ardahan", "Igdir", 
    "Yalova", "Karabuk", "Kilis", "Osmaniye", "Duzce" 
};

// Constructor to initialize and load constraints
RoutePlanner::RoutePlanner(const std::string& distance_data, const std::string& priority_data, const std::string& restricted_data, int maxDistance)
    : maxDistance(maxDistance), totalDistanceCovered(0), numPriorityProvinces(0), numWeatherRestrictedProvinces(0) {

    map.loadDistanceData(distance_data);
    map.resetVisited();
    loadPriorityProvinces(priority_data);
    loadWeatherRestrictedProvinces(restricted_data);
}

// Load priority provinces from txt file to an array of indices
void RoutePlanner::loadPriorityProvinces(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    std::string line;
    numPriorityProvinces = 0;
    while (std::getline(file, line) && numPriorityProvinces < MAX_PRIORITY_PROVINCES) {
        size_t pos = line.find('(');
        if (pos != std::string::npos) {
            std::string city = line.substr(0, pos - 1);
            int index = std::stoi(line.substr(pos + 1, line.find(')') - pos - 1));
            for (int i = 0; i < 81; ++i) {
                if (cities[i] == city) {
                    priorityProvinces[numPriorityProvinces++] = index;
                    break;
                }
            }
        }
    }
    file.close();
}

// Load weather-restricted provinces from txt file to an array of indices
void RoutePlanner::loadWeatherRestrictedProvinces(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    std::string line;
    numWeatherRestrictedProvinces = 0;
    while (std::getline(file, line) && numWeatherRestrictedProvinces < MAX_WEATHER_RESTRICTED_PROVINCES) {
        size_t pos = line.find('(');
        if (pos != std::string::npos) {
            std::string city = line.substr(0, pos - 1);
            int index = std::stoi(line.substr(pos + 1, line.find(')') - pos - 1));
            for (int i = 0; i < 81; ++i) {
                if (cities[i] == city) {
                    weatherRestrictedProvinces[numWeatherRestrictedProvinces++] = index;
                    break;
                }
            }
        }
    }
    file.close();
}

// Checks if a province is a priority province
bool RoutePlanner::isPriorityProvince(int province) const {
    for (int i = 0; i < numPriorityProvinces; ++i) {
        if (priorityProvinces[i] == province) {
            return true;
        }
    }
    return false;
}

// Checks if a province is weather-restricted
bool RoutePlanner::isWeatherRestricted(int province) const {
    for (int i = 0; i < numWeatherRestrictedProvinces; ++i) {
        if (weatherRestrictedProvinces[i] == province) {
            return true;
        }
    }
    return false;
}

// Begins the route exploration from the starting point
void RoutePlanner::exploreRoute(int startingCity) {
    map.markAsVisited(startingCity);
    stack.push(startingCity);  // Track the start in the stack for potential backtracking
    route.push_back(startingCity);  // Record the starting city in the route
    totalDistanceCovered = 0;  // Initialize distance covered

    exploreFromProvince(startingCity);  // Start the exploration from the starting city

    displayResults();  // Display the final results after the exploration is complete
}


void RoutePlanner::exploreFromProvince(int province) {
    // std::cout << "Exploring from province: " << cities[province] << std::endl;
    
    enqueueNeighbors(province);  // Enqueue neighbors of the current province

    while (!queue.isEmpty()) {
        int nextProvince = queue.dequeue();  // Dequeue the next province to explore
        // std::cout << "Dequeueing province: " << cities[nextProvince] << std::endl;

        if (isWeatherRestricted(nextProvince)) {
            std::cout << "Province " << cities[nextProvince] << " is weather-restricted. Skipping." << std::endl;
            continue;
        }

        if (!map.isVisited(nextProvince)) {
            int distance = map.getDistance(province, nextProvince);
            if (distance != -1 && distance <= maxDistance) {
                map.markAsVisited(nextProvince);
                stack.push(nextProvince);
                route.push_back(nextProvince);
                totalDistanceCovered += distance;
                // std::cout << "Visiting province: " << cities[nextProvince] << " (distance: " << distance << " km)" << std::endl;
                exploreFromProvince(nextProvince);  // Explore further from the next province
            }
        }
    }

    if (!isExplorationComplete()) {
        backtrack();  // Backtrack if the exploration is not complete
        // After backtracking, continue exploration from the last valid province
        if (!stack.isEmpty()) {
            int previousProvince = stack.peek();
            exploreFromProvince(previousProvince);  // Continue exploration from the previous province
        }
    }
}

void RoutePlanner::backtrack() {
    if (!stack.isEmpty()) {
        stack.pop();  // Pop the last province from the stack

        if (stack.isEmpty()) {
            return;  // If the stack is empty, no more backtracking possible
        }

        // The exploration will continue after popping from the stack
    }
}


void RoutePlanner::enqueueNeighbors(int province) {
    for (int i = 0; i < 81; ++i) {
        if (i != province && !map.isVisited(i)) {
            if (map.isWithinRange(province, i, maxDistance) && !isWeatherRestricted(i)) {
                if (isPriorityProvince(i)) {
                    queue.enqueuePriority(i);  // Enqueue with priority if it is a priority province
                } else {
                    queue.enqueue(i);  // Enqueue normally for other provinces
                }
            }
        }
    }
}


bool RoutePlanner::isExplorationComplete() const {
    // Check if stack and queue are both empty
    if (stack.isEmpty() && queue.isEmpty()) {
        return true;
    }

    // Check if no unvisited neighbors within range
    for (int i = 0; i < MAX_SIZE; ++i) {
        if (!map.isVisited(i)) {  // If province i has not been visited
            bool hasUnvisitedNeighbor = false;
            for (int j = 0; j < MAX_SIZE; ++j) {
                if (!map.isVisited(j) && map.isWithinRange(i, j, maxDistance)) {
                    hasUnvisitedNeighbor = true;
                    break;
                }
            }
            if (hasUnvisitedNeighbor) {
                return false;  // Still some unvisited neighbor within range
            }
        }
    }

    return true;  // No unvisited neighbors within range
}


void RoutePlanner::displayResults() const {
    std::cout << "----------------------------" << std::endl;
    std::cout << "Journey Completed!" << std::endl;
    std::cout << "----------------------------" << std::endl;
    std::cout << "Total Number of Provinces Visited: " << route.size() << std::endl;
    std::cout << "Total Distance Covered: " << totalDistanceCovered << " km" << std::endl;
    
    std::cout << "Route Taken: " << std::endl;
    for (size_t i = 0; i < route.size(); ++i) {
        std::cout << cities[route[i]];
        if (i < route.size() - 1) {
            std::cout << " -> ";
        }
    }
    std::cout << std::endl;

    // Priority Province Status
    std::cout << "Priority Provinces Status:" << std::endl;
    int totalVisitedPriority = 0;
    for (int i = 0; i < numPriorityProvinces; ++i) {
        std::string visitStatus = map.isVisited(priorityProvinces[i]) ? "Visited" : "Not Visited";
        std::cout << "- " << cities[priorityProvinces[i]] << " (" << visitStatus << ")" << std::endl;
        if (map.isVisited(priorityProvinces[i])) {
            totalVisitedPriority++;
        }
    }
    std::cout << "Total Priority Provinces Visited: " << totalVisitedPriority << " out of " << numPriorityProvinces << std::endl;

    // Success message for all visited priority provinces
    if (totalVisitedPriority == numPriorityProvinces) {
        std::cout << "Success: All priority provinces were visited." << std::endl;
    } else {
        std::cout << "Warning: Not all priority provinces were visited." << std::endl;
    }
}