#include <iostream>
#include "Isle.h"
#include "GameWorld.h"
#include "RealmShaper.h"

#define NUMBER_OF_INPUTS 3
#define NUMBER_OF_OUTPUTS 2

int main(int argc, char **argv)
{

    if (argc < (NUMBER_OF_INPUTS + NUMBER_OF_OUTPUTS + 1))
    {
        std::cerr << "[Main] " << "Not enough arguments" << std::endl;
    }

    std::string placesFile = argv[1];
    std::string playersFile = argv[2];

    std::vector<Isle *> places = Isle::readFromFile(placesFile);
    std::vector<RealmShaper *> players = RealmShaper::readFromFile(playersFile);
    GameWorld gameWorld = GameWorld();
    gameWorld.initializeGame(places, players);
    gameWorld.displayGameState();
    gameWorld.processGameEvents(argv[3]);
    gameWorld.saveGameState(argv[4], argv[5]);
}