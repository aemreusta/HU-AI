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
        std::istringstream ss(line);
        std::string city;
        while (std::getline(ss, city, ',')) {
            for (int i = 0; i < 81; ++i) {
                if (cities[i] == city) {
                    priorityProvinces[numPriorityProvinces++] = i;
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
        std::istringstream ss(line);
        std::string city;
        while (std::getline(ss, city, ',')) {
            for (int i = 0; i < 81; ++i) {
                if (cities[i] == city) {
                    weatherRestrictedProvinces[numWeatherRestrictedProvinces++] = i;
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
    stack.push(startingCity);
    route.push_back(startingCity);

    exploreFromProvince(startingCity);

    displayResults();
}

// Helper function to explore from a specific province
void RoutePlanner::exploreFromProvince(int province) {
    enqueueNeighbors(province);

    while (!queue.isEmpty()) {
        int nextProvince = queue.dequeue();
        if (!map.isVisited(nextProvince) && !isWeatherRestricted(nextProvince)) {
            int distance = map.getDistance(province, nextProvince);
            if (distance != -1 && totalDistanceCovered + distance <= maxDistance) {
                map.markAsVisited(nextProvince);
                stack.push(nextProvince);
                route.push_back(nextProvince);
                totalDistanceCovered += distance;
                exploreFromProvince(nextProvince);
            }
        }
    }

    if (!isExplorationComplete()) {
        backtrack();
    }
}

void RoutePlanner::enqueueNeighbors(int province) {
    for (int i = 0; i < 81; ++i) {
        if (i != province && map.isWithinRange(province, i, maxDistance) && !map.isVisited(i)) {
            if (isPriorityProvince(i)) {
                queue.enqueuePriority(i);
            } else {
                queue.enqueue(i);
            }
        }
    }
}

void RoutePlanner::backtrack() {
    if (!stack.isEmpty()) {
        int lastProvince = stack.pop();
        if (!stack.isEmpty()) {
            int previousProvince = stack.peek();
            totalDistanceCovered -= map.getDistance(previousProvince, lastProvince);
            exploreFromProvince(previousProvince);
        }
    }
}

bool RoutePlanner::isExplorationComplete() const {
    for (int i = 0; i < numPriorityProvinces; ++i) {
        if (!map.isVisited(priorityProvinces[i])) {
            return false;
        }
    }
    return true;
}

void RoutePlanner::displayResults() const {
    std::cout << "Journey Completed!" << std::endl;
    std::cout << "Total Provinces Visited: " << route.size() << std::endl;
    std::cout << "Total Distance Covered: " << totalDistanceCovered << " km" << std::endl;
    std::cout << "Route: ";
    for (int province : route) {
        std::cout << cities[province] << " ";
    }
    std::cout << std::endl;

    std::cout << "Priority Province Summary:" << std::endl;
    for (int i = 0; i < numPriorityProvinces; ++i) {
        std::cout << cities[priorityProvinces[i]] << ": " << (map.isVisited(priorityProvinces[i]) ? "Visited" : "Not Visited") << std::endl;
    }
}