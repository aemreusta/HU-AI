#ifndef REALMSHAPER_H
#define REALMSHAPER_H

#include <string>
#include <vector>
#include "Isle.h"

class RealmShaper
{
private:
    const std::string name;        // Name of the player
    int collectedEnergyPoints = 0; // Energy collected from Items
public:
    RealmShaper(std::string name); // Constructor declaration

    // Getters
    const std::string &getName() const; // Name getter
    int getEnergyLevel();               // Energy getter

    void collectItem(Item item); // Collect energy from item
    void loseEnergy();           // Energy is lost after Isle crafting
    bool hasEnoughEnergy();      // Checks if player has energy for Isle crafting

    // Overloaded operators
    bool operator==(const RealmShaper &other) const;
    friend std::ostream &operator<<(std::ostream &os, const RealmShaper &p);

    // Player parser
    static std::vector<RealmShaper *> readFromFile(const std::string &filename);
};

#endif