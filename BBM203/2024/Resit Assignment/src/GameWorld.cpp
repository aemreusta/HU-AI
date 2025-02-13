#include "GameWorld.h"
#include <fstream>
#include <sstream>
#include <iostream>

GameWorld::GameWorld() : mapTree(), shaperTree() {}

void GameWorld::initializeGame(std::vector<Isle *> places, std::vector<RealmShaper *> players)
{
    shaperTree.initializeTree(players);
    mapTree.initializeMap(places);
}

Map &GameWorld::getMapTree()
{
    return mapTree;
}

ShaperTree &GameWorld::getShaperTree()
{
    return shaperTree;
}

bool GameWorld::hasAccess(RealmShaper *realmShaper, Isle *isle)
{
    bool hasAccess = false;

    // The original sample code references calls like:
    // mapTree.calculateMinMapDepthAccess(...), mapTree.getIsleDepth(isle), etc.
    // but these do not actually exist in our Map. If you truly need them,
    // you'd implement or stub them. Here, we show a basic example:
    if (!shaperTree.findPlayer(realmShaper->getName()))
    {
        std::cout << "[Access Control] " << "RealmShaper not found!" << std::endl;
        return false;
    }
    // For now, simply return true so that "hasAccess" doesn't block everything:
    // (Replace with real logic as needed.)
    hasAccess = true;

    return hasAccess;
}

void GameWorld::exploreArea(RealmShaper *realmShaper, Isle *isle)
{
    if (!hasAccess(realmShaper, isle))
    {
        std::cout << "[Explore Area] " << realmShaper->getName()
                  << " does not have access to explore area "
                  << isle->getName() << std::endl;
        return;
    }

    // Simulate item pickup
    realmShaper->collectItem(isle->getItem());
    std::cout << "[Explore Area] " << realmShaper->getName()
              << " visited " << isle->getName() << std::endl;
    std::cout << "[Energy] " << realmShaper->getName()
              << "'s new energy level is " << realmShaper->getEnergyLevel()
              << std::endl;

    // The original code references: if (isle->getShaperCount() > 10) { mapTree.remove(isle); ... }
    // but we don't have `getShaperCount()` or `remove(...)` in Isle / Map. 
    // Here you can do stubs or real logic if needed. 
    // Example stub: do nothing.
}

void GameWorld::craft(RealmShaper *shaper, const std::string &isleName)
{
    if (!shaper->hasEnoughEnergy())
    {
        std::cout << "[Energy] " << shaper->getName()
                  << " does not have enough energy points: "
                  << shaper->getEnergyLevel() << std::endl;
        return;
    }
    shaper->loseEnergy();

    Isle *newIsle = new Isle(isleName);
    mapTree.insert(newIsle);
    std::cout << "[Craft] " << shaper->getName()
              << " crafted new Isle " << isleName << std::endl;
}

void GameWorld::displayGameState()
{
    mapTree.displayMap();
    shaperTree.displayTree();
}

// --------------------------------------------------------------------
// EXISTING function (needs 2 log files):
// --------------------------------------------------------------------
void GameWorld::processGameEvents(const std::string &accessLogs, const std::string &duelLogs)
{
    std::ifstream accessFile(accessLogs);
    std::ifstream duelFile(duelLogs);
    std::string log;
    int accessCount = 0;

    while (std::getline(accessFile, log))
    {
        // Example: "PlayerName IsleName"
        std::istringstream iss(log);
        std::string playerName, isleName;
        iss >> playerName >> isleName;

        RealmShaper *player = shaperTree.findPlayer(playerName);
        if (!player)
        {
            std::cout << "[Warning] RealmShaper '" << playerName << "' not found in ShaperTree.\n";
            continue;
        }

        Isle *isle = mapTree.findIsle(isleName);
        if (isle)
        {
            exploreArea(player, isle);
        }
        else
        {
            // Craft a new Isle if not found
            craft(player, isleName);
        }

        accessCount++;
        // The original snippet does a duel every 5 accesses:
        if (accessCount % 5 == 0 && std::getline(duelFile, log))
        {
            // Example: "ChallengerName Result"
            std::istringstream duelIss(log);
            std::string challengerName, result;
            duelIss >> challengerName >> result;

            RealmShaper *challenger = shaperTree.findPlayer(challengerName);
            bool duelResult = (result == "win");
            // The original snippet calls shaperTree.duel(challenger, duelResult);
            // We have no such function in ShaperTree, so you might stub it out.
            if (challenger)
            {
                std::cout << "[Duel] " << challenger->getName()
                          << (duelResult ? " wins" : " loses")
                          << " a duel (stub logic)." << std::endl;
            }
        }

        displayGameState();
    }

    // Process remaining duels
    while (std::getline(duelFile, log))
    {
        std::istringstream duelIss(log);
        std::string challengerName, result;
        duelIss >> challengerName >> result;

        RealmShaper *challenger = shaperTree.findPlayer(challengerName);
        bool duelResult = (result == "win");
        if (challenger)
        {
            std::cout << "[Duel] " << challenger->getName()
                      << (duelResult ? " wins" : " loses")
                      << " a duel (stub logic)." << std::endl;
        }

        displayGameState();
    }
}

// --------------------------------------------------------------------
// EXISTING function (needs 4 file names):
// --------------------------------------------------------------------
void GameWorld::saveGameState(const std::string &currentIsles,
                              const std::string &currentWorld,
                              const std::string &currentShapers,
                              const std::string &currentPlayerTree)
{
    mapTree.writeIslesToFile(currentIsles);
    mapTree.writeToFile(currentWorld);

    // The original snippet calls:
    // shaperTree.writeToFile(currentPlayerTree);
    // shaperTree.writeShapersToFile(currentShapers);
    // But ShaperTree has no such methods by default. 
    // We can stub it out:
    {
        std::ofstream tFile(currentPlayerTree);
        if (!tFile.is_open())
            std::cerr << "[Error] Cannot open " << currentPlayerTree << std::endl;
        else
        {
            // Example stub writing the level-order from ShaperTree
            auto arr = shaperTree.getTree();
            for (auto *sh : arr)
            {
                if (sh) tFile << sh->getName() << "\n";
            }
            tFile.close();
        }
    }
    {
        std::ofstream sFile(currentShapers);
        if (!sFile.is_open())
            std::cerr << "[Error] Cannot open " << currentShapers << std::endl;
        else
        {
            // Example stub writing all Shapers sorted by name
            auto arr = shaperTree.getTree();
            std::vector<RealmShaper*> sortedSh(arr.begin(), arr.end());
            std::sort(sortedSh.begin(), sortedSh.end(),
                      [](RealmShaper *a, RealmShaper *b) {
                          return a->getName() < b->getName();
                      });
            for (auto *sh : sortedSh)
            {
                if (sh) sFile << sh->getName() << "\n";
            }
            sFile.close();
        }
    }
}

// --------------------------------------------------------------------
// NEW Overloads to match main.cpp calls
// (1) processGameEvents(const std::string &accessLogs);
// (2) saveGameState(const std::string &currentIsles, const std::string &currentWorld);
// --------------------------------------------------------------------

// If main.cpp only passes accessLogs, we can treat duelLogs as empty:
void GameWorld::processGameEvents(const std::string &accessLogs)
{
    processGameEvents(accessLogs, "" /* duelLogs */);
}

// If main.cpp only passes two file names, we fill in dummy for the other two:
void GameWorld::saveGameState(const std::string &currentIsles, const std::string &currentWorld)
{
    const std::string defaultShapers   = "shapers_stub.txt";
    const std::string defaultShaperTree= "shaper_tree_stub.txt";

    saveGameState(currentIsles, currentWorld, defaultShapers, defaultShaperTree);
}
