#ifndef GAMEWORLD_H
#define GAMEWORLD_H

#include "Map.h"
#include "RealmShapers.h"
#include <string>
#include <vector>

class GameWorld
{
private:
    Map mapTree;
    ShaperTree shaperTree;

public:
    // Constructor declaration
    GameWorld();

    // Getters
    Map &getMapTree();
    ShaperTree &getShaperTree();

    // Initializes game by initializing the trees
    void initializeGame(std::vector<Isle *> isles, std::vector<RealmShaper *> realmShapers);

    // Checks access for a realmShaper for an isle
    bool hasAccess(RealmShaper *realmShaper, Isle *isle);

    // Player explores existing area
    void exploreArea(RealmShaper *realmShaper, Isle *isle);

    // Player crafts non-existing Isle
    void craft(RealmShaper *shaper, const std::string &isleName);

    // Displays game state in terminal
    void displayGameState();

    // --------------------------------------------------------------------
    // Existing method that requires 2 log files:
    // --------------------------------------------------------------------
    void processGameEvents(const std::string &accessLogs, const std::string &duelLogs);

    // --------------------------------------------------------------------
    // Existing method that requires 4 file names:
    // --------------------------------------------------------------------
    void saveGameState(const std::string &currentIsles,
                       const std::string &currentWorld,
                       const std::string &currentShapers,
                       const std::string &currentPlayerTree);

    // --------------------------------------------------------------------
    // NEW Overloads (to match calls in main.cpp):
    // --------------------------------------------------------------------
    // 1) Overload taking ONLY the access log
    void processGameEvents(const std::string &accessLogs);

    // 2) Overload taking ONLY two output files
    void saveGameState(const std::string &currentIsles, const std::string &currentWorld);
};

#endif
