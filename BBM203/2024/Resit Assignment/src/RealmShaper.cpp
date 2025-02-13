#include "RealmShaper.h"
#include <fstream>
#include <sstream>
#include <iostream>

#define NECESSARY_ENERGY 2000 // Necessary energy to craft an Isle

RealmShaper::RealmShaper(std::string name) : name(name)
{
}

const std::string &RealmShaper::getName() const
{
    return this->name;
}

int RealmShaper::getEnergyLevel()
{
    return this->collectedEnergyPoints;
}

void RealmShaper::collectItem(Item item)
{
    this->collectedEnergyPoints += item;
}

void RealmShaper::loseEnergy()
{
    this->collectedEnergyPoints -= NECESSARY_ENERGY;
}

bool RealmShaper::hasEnoughEnergy()
{
    return this->collectedEnergyPoints >= NECESSARY_ENERGY;
}

std::vector<RealmShaper *> RealmShaper::readFromFile(const std::string &filename)
{
    std::vector<RealmShaper *> players;
    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return players;
    }

    std::string line;
    while (std::getline(file, line))
    {
        // Skip empty lines or comment lines
        if (line.empty() || line[0] == '#')
            continue;

        std::istringstream iss(line);
        std::string playerName;
        int honourPoints = 0;

        // Parse first token as playerName
        if (!(iss >> playerName))
            continue; // no valid name found

        // Attempt to parse second token as honourPoints (optional)
        if (!(iss >> honourPoints))
            honourPoints = 0; // default if missing

        RealmShaper *player = new RealmShaper(playerName);
        player->collectedEnergyPoints = honourPoints;
        players.push_back(player);
    }

    file.close();
    return players;
}

bool RealmShaper::operator==(const RealmShaper &other) const
{
    // Compare by name
    return (this->name == other.name);
}

std::ostream &operator<<(std::ostream &os, const RealmShaper &p)
{
    return (os << p.name);
}
