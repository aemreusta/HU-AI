#include "GameWorld.h"

GameWorld::GameWorld() : mapTree(), shaperTree() {}

void GameWorld::initializeGame(std::vector<Isle *> places, std::vector<RealmShaper *> players)
{
    shaperTree.initializeTree(players);
    mapTree.initializeMap(places);
}

Map& GameWorld::getMapTree()
{
    return mapTree;
}
ShaperTree& GameWorld::getShaperTree()
{
    return shaperTree;
}

bool GameWorld::hasAccess(RealmShaper *realmShaper, Isle *isle)
{
    bool hasAccess = false;

    // TODO: Check if the realmShaper has access to explore the isle
    // Get necessary depth values
    // Use mapTree.calculateMinMapDepthAccess
    // Use // std::cout << "[Access Control] " << "RealmShaper not found!" << std::endl;

    if (!shaperTree.findPlayer(realmShaper->getName())) {
        std::cout << "[Access Control] " << "RealmShaper not found!" << std::endl;
        return false;
    }
    int shaperDepth = shaperTree.getDepth(realmShaper);
    int requiredDepth = mapTree.calculateMinMapDepthAccess(mapTree.getIsleDepth(isle), shaperTree.getDepth(), mapTree.getDepth());
    return shaperDepth >= requiredDepth;

    return hasAccess;
}

void GameWorld::exploreArea(RealmShaper *realmShaper, Isle *isle)
{
    // TODO:
    // Check if realmShaper has access
    // Use // std::cout << "[Explore Area] " << realmShaper->getName() << " does not have access to explore area " << *isle << std::endl;
    // If realmShaper has access
    // Visit isle, 
    // collect item, 
    // check overcrowding for Isle, 
    // delete Isle if necessary

    // Use // std::cout << "[Explore Area] " << realmShaper->getName() << " visited " << isle->getName() << std::endl;
    // Use // std::cout << "[Energy] " << realmShaper->getName() << "'s new energy level is " << realmShaper->getEnergyLevel() << std::endl;
    // Use // std::cout << "[Owercrowding] " << isle->getName() << " self-destructed, it will be removed from the map" << std::endl;

    // You will need to implement a mechanism to keep track of how many realm shapers are at an Isle at the same time
    // There are more than one ways to do this, so it has been left completely to you
    // Use shaperCount, but that alone will not be enough,
    // you will likely need to add attributes that are not currently defined
    // to RealmShaper or Isle or other classes depending on your implementation

    if (!hasAccess(realmShaper, isle)) {
        std::cout << "[Explore Area] " << realmShaper->getName() << " does not have access to explore area " << isle->getName() << std::endl;
        return;
    }
    realmShaper->collectItem(isle->getItem());
    std::cout << "[Explore Area] " << realmShaper->getName() << " visited " << isle->getName() << std::endl;
    std::cout << "[Energy] " << realmShaper->getName() << "'s new energy level is " << realmShaper->getEnergyLevel() << std::endl;

    if (isle->getShaperCount() > 10) {
        mapTree.remove(isle);
        std::cout << "[Overcrowding] " << isle->getName() << " self-destructed, it will be removed from the map" << std::endl;
    }
}

void GameWorld::craft(RealmShaper *shaper, const std::string &isleName){
    // TODO: Check energy and craft new isle if possible
    // Use std::cout << "[Energy] " << shaperName << " has enough energy points: " << shaperEnergyLevel << std::endl;
    // Use std::cout << "[Craft] " << shaperName << " crafted new Isle " << isleName << std::endl;
    // Use std::cout << "[Energy] " << shaperName << " does not have enough energy points: " << shaperEnergyLevel << std::endl;
    if (!shaper->hasEnoughEnergy()) {
        std::cout << "[Energy] " << shaper->getName() << " does not have enough energy points: " << shaper->getEnergyLevel() << std::endl;
        return;
    }
    shaper->loseEnergy();
    Isle *newIsle = new Isle(isleName);
    mapTree.insert(newIsle);
    std::cout << "[Craft] " << shaper->getName() << " crafted new Isle " << isleName << std::endl;
}

void GameWorld::displayGameState()
{
    mapTree.displayMap();
    shaperTree.displayTree();
}

// TODO: Implement functions to read and parse Access and Duel logs

void GameWorld::processGameEvents(const std::string &accessLogs, const std::string &duelLogs)
{
    // TODO:
    // Read logs
    // For every 5 access, 1 duel happens
    // If there are still duel logs left after every access happens duels happens one after other

    // This function should call exploreArea and craft functions

    // Use displayGameState();

    std::ifstream accessFile(accessLogs);
    std::ifstream duelFile(duelLogs);
    std::string log;
    int accessCount = 0;

    while (std::getline(accessFile, log)) {
        // Parse access log and handle exploration or crafting
        // Example log: "PlayerName IsleName"
        std::istringstream iss(log);
        std::string playerName, isleName;
        iss >> playerName >> isleName;

        RealmShaper *player = shaperTree.findPlayer(playerName);
        Isle *isle = mapTree.findIsle(isleName);

        if (isle) {
            exploreArea(player, isle);
        } else {
            craft(player, isleName);
        }

        accessCount++;
        if (accessCount % 5 == 0 && std::getline(duelFile, log)) {
            // Parse duel log and process duel
            // Example log: "ChallengerName OpponentName Result"
            std::istringstream duelIss(log);
            std::string challengerName, result;
            duelIss >> challengerName >> result;

            RealmShaper *challenger = shaperTree.findPlayer(challengerName);
            bool duelResult = (result == "win");
            shaperTree.duel(challenger, duelResult);
        }

        displayGameState();
    }

    // Process remaining duels
    while (std::getline(duelFile, log)) {
        std::istringstream duelIss(log);
        std::string challengerName, result;
        duelIss >> challengerName >> result;

        RealmShaper *challenger = shaperTree.findPlayer(challengerName);
        bool duelResult = (result == "win");
        shaperTree.duel(challenger, duelResult);
        displayGameState();
    }
}

void GameWorld::saveGameState(const std::string &currentIsles, const std::string &currentWorld, const std::string &currentShapers, const std::string &currentPlayerTree)
{
    mapTree.writeIslesToFile(currentIsles);
    mapTree.writeToFile(currentWorld);
    shaperTree.writeToFile(currentPlayerTree);
    shaperTree.writeShapersToFile(currentShapers);
}